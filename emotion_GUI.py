import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import threading
import pygame
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_spoken_emotion = None  # to prevent repeating the same emotion

# Load face detection model and emotion model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('emotion_model.h5')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# GUI setup
root = tk.Tk()
root.title("Anime-Themed Emotion Detector")
root.geometry("900x600")
root.resizable(False, False)

# Load and set anime-style background
bg_image = Image.open("8-bit-graphics-pixels-scene-with-person-sunset.jpg")
bg_image = bg_image.resize((900, 600), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Frame over background
frame = tk.Frame(root, bg="black", bd=2)
frame.place(x=100, y=80, width=700, height=450)

# Camera and emotion label
camera_label = Label(frame)
camera_label.pack()
emotion_label = Label(root, text="", font=("Helvetica", 20, "bold"), fg="white", bg="#000000", pady=10)
emotion_label.place(relx=0.5, rely=0.9, anchor="center")

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

def speak_emotion(emotion):
    global last_spoken_emotion
    if emotion != last_spoken_emotion:
        last_spoken_emotion = emotion
        if emotion == "No face detected":
            engine.say("Face not detected")
        else:
            engine.say(f"You are {emotion.lower()} now")
        engine.runAndWait()

# Emotion detection function
def detect_emotion():
    ret, frame = cap.read()
    if not ret:
        root.after(10, detect_emotion)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    emotion_text = "No face detected"
    detected_emotion = "No face detected"

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        label = class_labels[np.argmax(preds)]
        emotion_text = f"Emotion: {label}"
        detected_emotion = label

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        break  # process only one face

    # Convert and show camera image
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_image)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    emotion_label.configure(text=emotion_text)
    speak_emotion(detected_emotion)

    root.after(500, detect_emotion)  # update every 500 ms

# Anime background music

pygame.mixer.init()
pygame.mixer.music.load("anime_bgm.mp3")
pygame.mixer.music.set_volume(0.2)  # 🔉 Set background music to 20% volume
pygame.mixer.music.play(-1)


# Start detection
def start_detection():
    detect_emotion()

# Button to start
start_button = Button(root, text="Start Emotion Detection", font=("Helvetica", 14, "bold"), command=start_detection, bg="#FF69B4", fg="white")
start_button.place(relx=0.5, rely=0.05, anchor="center")

# Run GUI
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
