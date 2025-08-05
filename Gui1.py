import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import threading
import time
import pygame
import pyttsx3

# Load model and labels
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('emotion_model.h5')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# GUI setup
root = tk.Tk()
root.title("Real-Time Emotion Detector")
root.geometry("900x600")
root.resizable(False, False)

# Background image
bg_image = Image.open("8-bit-graphics-pixels-scene-with-person-sunset.jpg")
bg_image = bg_image.resize((900, 600), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Frame and label setup
frame = tk.Frame(root, bg="black", bd=2)
frame.place(x=100, y=80, width=700, height=450)
camera_label = Label(frame)
camera_label.pack()
emotion_label = Label(root, text="", font=("Helvetica", 20, "bold"), fg="white", bg="#000000", pady=10)
emotion_label.place(relx=0.5, rely=0.9, anchor="center")

# Initialize camera
cap = cv2.VideoCapture(0)
last_frame_time = time.time()
last_spoken = ""
last_spoken_time = 0
frame_counter = 0
current_image = None
current_emotion = "No face detected"
stop_flag = False

# Text to speech
engine = pyttsx3.init()
engine.setProperty('rate', 130)

# Background music
pygame.mixer.init()
pygame.mixer.music.load("anime_bgm.mp3")
pygame.mixer.music.set_volume(0.1)
pygame.mixer.music.play(-1)

def speak_emotion(emotion):
    global last_spoken, last_spoken_time
    if emotion != last_spoken and (time.time() - last_spoken_time) > 2:
        engine.say(f"You are {emotion.lower()}" if emotion != "No face detected" else "Face not detected")
        engine.runAndWait()
        last_spoken = emotion
        last_spoken_time = time.time()

def process_frame():
    global current_image, current_emotion, frame_counter

    ret, frame = cap.read()
    if not ret:
        return

    frame_counter += 1
    if frame_counter % 3 != 0:
        return  # skip every 2 out of 3 frames to save processing time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    emotion = "No face detected"
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = model.predict(roi, verbose=0)[0]
        label = class_labels[np.argmax(preds)]
        emotion = label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        break  # Process only one face for speed

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    current_image = ImageTk.PhotoImage(image=img)
    current_emotion = emotion

    speak_emotion(emotion)

def update_gui():
    if current_image:
        camera_label.configure(image=current_image)
        camera_label.image = current_image
        emotion_label.configure(text=f"Emotion: {current_emotion}")
    if not stop_flag:
        root.after(30, update_gui)  # Schedule GUI update every 30 ms

def video_loop():
    while not stop_flag:
        process_frame()
        time.sleep(0.03)  # ~30 FPS for processing thread

def start_detection():
    threading.Thread(target=video_loop, daemon=True).start()
    update_gui()

def on_closing():
    global stop_flag
    stop_flag = True
    cap.release()
    pygame.mixer.music.stop()
    root.destroy()

start_button = Button(root, text="Start Emotion Detection", font=("Helvetica", 14, "bold"),
                      command=start_detection, bg="#FF69B4", fg="white")
start_button.place(relx=0.5, rely=0.05, anchor="center")
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()

