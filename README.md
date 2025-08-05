# facedetectionsystem


## 💡 **Project Title**: Face Detection System – Emotion Recognition using Deep Learning

### 🎯 **Objective:**

To build a deep learning model that detects human facial emotions in real time and classifies them into categories like **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.

---

### ⚙️ **Technology Stack:**

* **Languages**: Python
* **Libraries**: OpenCV, TensorFlow / Keras, NumPy, Matplotlib
* **Model Type**: Convolutional Neural Network (CNN)
* **Dataset**: FER-2013 (Facial Expression Recognition dataset)
* **Tools**: Jupyter Notebook / Google Colab

---

### 🧠 **Model Architecture:**

* **Input**: 48x48 grayscale face images
* **Layers**:

  * Convolutional layers + ReLU
  * MaxPooling layers
  * Dropout (to prevent overfitting)
  * Fully Connected Layers
  * Softmax activation for 7-class output

---

### 🔍 **Emotions Detected:**

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

---

### 📈 **Model Performance:**

* **Accuracy**: \~65–75% on validation/test set
* **Loss Function**: Categorical Crossentropy
* **Optimizer**: Adam
* **Evaluation Metrics**: Accuracy, Confusion Matrix

---

### 🖥️ **Features:**

* Real-time webcam emotion detection using OpenCV
* Bounding box around the detected face
* Live emotion label display over the face
* Option to save emotion data logs for analysis

---

### 🌐 **Use Cases:**

* Classroom/student engagement monitoring
* Mental health applications
* Customer sentiment analysis
* Human-computer interaction (HCI)

