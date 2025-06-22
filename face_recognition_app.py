import face_recognition
import cv2
import numpy as np
import sys
import threading
import time
import requests
import math
import requests 
import getpass
import websocket
import json

SYSTEM_NAME = "Sistem 1"

# de rezolvat: 
# - (Rezolvat-ish) nu merge timerul
# - (Rezolvat) se fac mai multe requesturi in mai multe frame uri
# - Se fac requesturi o data la 20 de secunde chiar daca nu e nicio fata in frame

#ready_to_request = True
ready_to_request = threading.Event()
ready_to_enter_else = threading.Event()
system_stopped_event = threading.Event()
cached_names = []
cached_encodings = []

# def request_timer():
#     global ready_to_request
#     while True:
#         if not ready_to_request:
#             time.sleep(20)
#             ready_to_request = True
#         else:
#             time.sleep(1)

def login_and_start():
    print(f"Welcome to system {SYSTEM_NAME}")
    while True:
        password = getpass.getpass("Enter password: ")
        try:
            response = requests.post(
                "http://localhost:8000/recognition-login",
                json={"name": SYSTEM_NAME, "password": password}
            )
            if response.status_code == 200 and response.json().get("success"):
                system_id = response.json().get("system_id")
                system_token = response.json().get("system_token")
                lock_password = response.json().get("lock_password")
                print("Login successful.")
                start_face_recognition(system_id, system_token, lock_password)
                return
            else:
                print("Incorrect password. Please try again.")
        except Exception as e:
            print("Error connecting to backend:", e)
            time.sleep(2)

def listen_for_updates(system_id, state_event, logout_event, start_event, stop_event):
    def on_open(ws):
        ws.send(json.dumps({"system_id": system_id}))

    def on_message(ws, message):
        data = json.loads(message)
        if data.get("event") == "update_encodings":
            print("Received update signal, refreshing encodings...")
            cached_names.clear()
            cached_encodings.clear()
            ready_to_request.set()
        elif data.get("event") == "system_stopped":
            print("System stopped signal received. Entering sleep mode.")
            state_event.set()
        elif data.get("event") == "system_started":
            print("System started signal received. Waking up.")
            start_event.set()
        elif data.get("event") == "system_logout":
            print("Logout signal received. Returning to login.")
            logout_event.set()

    ws = websocket.WebSocketApp(
        "ws://localhost:8000/ws/updates",
        on_open=on_open,
        on_message=on_message
    )

    # Run the websocket in a loop that checks stop_event
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    while not stop_event.is_set():
        time.sleep(0.1)
    ws.close()

def sleeping_state(state_event, start_event, logout_event, system_id):
    dot_count = 0
    while True:
        if logout_event.is_set():
            logout_event.clear()
            return "logout"
        if start_event.is_set():
            start_event.clear()
            return "start"
        print("System is closed" + "." * (dot_count + 1))
        dot_count = (dot_count + 1) % 5
        time.sleep(5)

def notify_backend(status, system_id, name=None, system_token=None):
    data = {'status': status}
    if name:
        data['name'] = name
    headers = {}
    if system_token:
        headers['Authorization'] = f'Bearer {system_token}'
    try:
        requests.post(f"http://localhost:8000/notify/{system_id}", json=data, headers=headers)
    except Exception as e:
        print("Failed to notify backend: ", e)

def notify_lock(unlock: bool, lock_password: str):
    """
    Sends a GET request to the lock controller.
    unlock=True: unlocks the door
    unlock=False: locks the door
    The lock_password is sent in the Authorization header.
    """
    action = "off" if unlock else "on"
    url = f"http://192.168.1.135/5/{action}"
    headers = {"Authorization": lock_password}
    try:
        requests.get(url, headers=headers, timeout=2)
    except Exception as e:
        print("Failed to notify lock: ", e)

# Function to update cached face encodings from an API
def update_cached_encodings(system_id):
    global cached_encodings, cached_names, ready_to_request
    while True:
        if ready_to_request.is_set():
            try:
                print("Making API request to update cached encodings...")
                response = requests.get(f"http://localhost:8000/{system_id}/encodings")
                if response.status_code == 200:
                    json_response = response.json()
                    cached_names.clear()
                    cached_encodings.clear()
                    for item in json_response:
                        # print(item)
                        cached_names.append(item['firstName'])
                        # If you store encoding as a string in the DB, convert it to a list/array here
                        # For example, if it's a JSON string: encoding = json.loads(item['encoding'])
                        # If it's already a list: encoding = item['encoding']
                        encoding = item.get('encoding')
                        if encoding is not None:
                            cached_encodings.append(encoding)
                    print("Cached encodings updated.")
                else:
                    print("Failed to fetch encodings:", response.status_code)
            except Exception as e:
                print("Error updating encodings:", e)
            finally:
                ready_to_request.clear()
                time.sleep(20)
                ready_to_enter_else.set()



def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
        
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5)*2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def start_face_recognition(system_id, system_token, lock_password):
    # Events for controlling state
    state_event = threading.Event()   # Sleep mode
    logout_event = threading.Event()  # Logout to login
    start_event = threading.Event()   # Wake up from sleep
    stop_event = threading.Event()

    listener_thread = threading.Thread(
        target=listen_for_updates,
        args=(system_id, state_event, logout_event, start_event, stop_event),
        daemon=True
    )
    listener_thread.start()


    process_current_frame = True
    face_locations = []
    face_encodings = []
    face_names = []
    global cached_encodings
    global ready_to_request
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        sys.exit('Video source not found...')

    threading.Thread(target=update_cached_encodings, args=(system_id,), daemon=True).start()
    ready_to_enter_else.set()

    consecutive_frames_required = 5
    recognized_counter = 0
    unknown_counter = 0
    last_recognized_name = None
    authenticated = False
    unknown_notified = False
    reset_frames_required = 10  # Number of frames with no face to reset authorization
    not_detected_counter = 0


    while True:
        if logout_event.is_set():
            print("Logging out and returning to login...")
            break
        if state_event.is_set():
            print("Stopping face recognition and entering sleep mode...")
            video_capture.release()
            cv2.destroyAllWindows()
            # Sleep until system_started or logout
            result = sleeping_state(state_event, start_event, logout_event, system_id)
            if result == "logout":
                break
            # Re-initialize video capture after waking up
            video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                sys.exit('Video source not found...')
            ready_to_enter_else.set()
            state_event.clear()
            continue

        ret, frame = video_capture.read()
        if not ret:
            continue

        if process_current_frame:
            name = 'Unknown'
            confidence = 'Unknown'
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

             # --- Reset logic: if no face detected, increment counter ---
            if len(face_encodings) == 0:
                not_detected_counter += 1
                if not_detected_counter >= reset_frames_required:
                    authenticated = False
                    last_recognized_name = None
                    recognized_counter = 0
            else:
                not_detected_counter = 0  # Reset counter if any face is detected

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(cached_encodings, face_encoding)
                if any(matches):
                    face_distances = face_recognition.face_distance(cached_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    name = cached_names[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])

                    if name == last_recognized_name:
                        recognized_counter += 1
                    else:
                        recognized_counter = 1
                        last_recognized_name = name

                    if recognized_counter >= consecutive_frames_required and not authenticated:
                        notify_backend("success", system_id, name, system_token=system_token)
                        notify_lock(unlock=True, lock_password=lock_password)
                        authenticated = True
                        unknown_counter = 0
                        unknown_notified = False
                else:
                    recognized_counter = 0
                    last_recognized_name = None
                    authenticated = False
                    if ready_to_enter_else.is_set():
                        unknown_counter += 1
                        if unknown_counter >= consecutive_frames_required and not unknown_notified:
                            ready_to_request.set()
                            ready_to_enter_else.clear()
                            notify_backend("fail", system_id, system_token=system_token)
                            notify_lock(unlock=False, lock_password=lock_password)
                            unknown_notified = True
                    else:
                        unknown_counter = 0
                        unknown_notified = False

                face_names.append(f'{name} ({confidence})')

        process_current_frame = not process_current_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    stop_event.set()
    time.sleep(0.2)
    video_capture.release()
    cv2.destroyAllWindows()
    system_stopped_event.clear()
    login_and_start()

if __name__ == '__main__':
    login_and_start()