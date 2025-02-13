import cv2 
import os 
import numpy as np 
from tensorflow.keras.models import model_from_json 
import face_recognition 
import csv 
from datetime import datetime 
# Load the database images and names 
database_dir = 'Images_Attendance' 
known_face_encodings = [] 
known_face_names = [] 
for filename in os.listdir(database_dir): 
 if filename.endswith(".jpg"): 
 name = os.path.splitext(filename)[0] 
 image_path = os.path.join(database_dir, filename) 
 image = face_recognition.load_image_file(image_path) 
 encoding = face_recognition.face_encodings(image)[0] 
 known_face_encodings.append(encoding) 
 known_face_names.append(name) 
# Load Face Detection Model 
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml") 
# Load Anti-Spoofing Model graph 
with open('antispoofing_models/antispoofing_model.json', 'r') as json_file: 
 loaded_model_json = json_file read() 
model = model_from_json(loaded_model_json) 
# Load anti-spoofing model weights 
model.load_weights('antispoofing_models/antispoofing_model.h5') 
print("Model loaded from disk") # Create or open the CSV file for attendance 
csv_file = 'Attendance.csv' 
if not os.path.exists(csv_file): 
 with open(csv_file, mode='w', newline='') as file: 
 writer = csv.writer(file) 
 writer.writerow(['Name', 'Date', 'Time', 'Real or Spoof']) 
# Initialize the detected_names set 
detected_names = set() 
video = cv2.VideoCapture(0) 
while True: 
 try: 
 ret, frame = video.read() 
 if not ret: 
 break # Exit loop if no frame captured 
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
 faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
 for (x, y, w, h) in faces: 
 face = frame[y-5:y+h+5, x-5:x+w+5] 
 resized_face = cv2.resize(face, (160, 160)) 
 resized_face = resized_face.astype("float") / 255.0 
 resized_face = np.expand_dims(resized_face, axis=0) 
 preds = model.predict(resized_face)[0] 
 # Face recognition 
 rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
 face_locations = face_recognition.face_locations(rgb_small_frame) 
 face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) for (top, right, bottom, left), encoding in zip(face_locations, face_encodings): 
 matches = face_recognition.compare_faces(known_face_encodings, encoding) 
 name = "Unknown" 
 if True in matches: 
 first_match_index = matches.index(True) 
 name = known_face_names[first_match_index] 
 # Update attendance in the CSV file only if not already detected 
 if name not in detected_names: 
 date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
 label = 'spoof' if preds > 0.5 else 'real' 
 with open(csv_file, mode='a', newline='') as file: 
 writer = csv.writer(file) 
 writer.writerow([name, date_time, label]) 
 detected_names.add(name) 
 label_color = (0, 0, 255) if preds > 0.5 else (0, 255, 0) 
 cv2.putText(frame, f"{name} ({'Spoof' if preds > 0.5 else 'Real'})", (x, y - 10), 
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2) 
 cv2.rectangle(frame, (x, y), (x+w, y+h), label_color, 2) 
 cv2.imshow('frame', frame) 
 key = cv2.waitKey(1) 
 if key == ord('q'): 
 break 
 except Exception as e: 
 pass 
video.release() 
cv2.destroyAllWindows()
