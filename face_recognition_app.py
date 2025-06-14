import face_recognition
import cv2
import numpy as np
import sys
import threading
import time
import requests
import math

# de rezolvat: 
# - (Rezolvat-ish) nu merge timerul
# - (Rezolvat) se fac mai multe requesturi in mai multe frame uri
# - Se fac requesturi o data la 20 de secunde chiar daca nu e nicio fata in frame

#ready_to_request = True
ready_to_request = threading.Event()
ready_to_enter_else = threading.Event()
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

# Function to update cached face encodings from an API
def update_cached_encodings():
    global cached_encodings, ready_to_request
    while True:
        if ready_to_request.is_set():
            try:
                print("Making API request to update cached encodings...")
                response = requests.get("http://localhost:8000/people/encodings")
                if response.status_code == 200:
                    # Adjust this parsing to match your actual API format
                    #print(response.json())
                    json_response = response.json()
                    #cached_encodings = [np.array(item['encoding']) for item in json_response]
                    for item in json_response:
                        # print(item)
                        cached_names.append(item['firstName'])
                        cached_encodings.append(item['encoding'])
                    #print(cached_encodings)
                    print("Cached encodings updated.")
                else:
                    print("Failed to fetch encodings:", response.status_code)
            except Exception as e:
                print("Error updating encodings:", e)
            finally:
                #ready_to_request = False
                ready_to_request.clear()
                #print("Ready to request in thread", ready_to_request)
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


def start_face_recognition():

    process_current_frame = True
    face_locations = []
    face_encodings = []
    face_names = []
    global cached_encodings
    global ready_to_request
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        sys.exit('Video source not found...')

    threading.Thread(target=update_cached_encodings, daemon=True).start()
    ready_to_enter_else.set()

    while True:
        #print(ready_to_request.is_set())
        ret, frame = video_capture.read()
       # print("Ready to request in main: ", ready_to_request)


        if process_current_frame:
            name = 'Unknown'
            confidence = 'Unknown'
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            #Find all faces in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                #print("Cached encodings in main: ", cached_encodings)
                matches = face_recognition.compare_faces(cached_encodings, face_encoding)
                # matches = [False, True, True, False]
                
                # if the encoding is in the cache memory
                if any(matches):
                    face_distances = face_recognition.face_distance(cached_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    name = cached_names[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])
                elif ready_to_enter_else.is_set():
                    # name = 'Unknown'
                    # confidence = 'Unknown'
                    #ready_to_request = True
                    ready_to_request.set()
                    ready_to_enter_else.clear()
                   

                face_names.append(f'{name} ({confidence})')

        process_current_frame = not process_current_frame

        # Display annotations
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

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_face_recognition()