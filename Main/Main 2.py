# %%
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

data_dir = r"C:\Users\Asus\OneDrive\Documents\Ip\EmotionsDS\EmotionsDS\Emotions Dataset\Emotions Dataset\test"  
emotion_folders = ['angry', 'happy', 'sad']  
emotion_labels = {'angry': 0, 'happy': 1, 'sad': 2}

images = []
labels = []

for emotion in emotion_folders:
    folder_path = os.path.join(data_dir, emotion)

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        
        if img is not None:
            img = cv2.resize(img, (48, 48))
            images.append(img)
            labels.append(emotion_labels[emotion])

images = np.array(images).reshape(-1, 48, 48, 1) 
images = images / 255.0  
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# %%
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()

model.add(layers.Input(shape=(48, 48, 1)))  

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  
model.add(layers.Dense(3, activation='softmax'))  

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# %%
history = model.fit(X_train, y_train, epochs=25, 
                    validation_split=0.1,  # Using part of the training set for validation
                    batch_size=64)


# %%
model.save('emotion_detection_model.keras')


# %%
model = tf.keras.models.load_model('emotion_detection_model.keras')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# %%
import os
import cv2
import numpy as np

test_data_dir = r"C:\Users\Asus\OneDrive\Documents\Ip\EmotionsDS\EmotionsDS\Emotions Dataset\Emotions Dataset\test"
emotion_folders = ['angry', 'happy', 'sad'] 

test_images = []
test_labels = []

emotion_labels = {'angry': 0, 'happy': 1, 'sad': 2}

for emotion in emotion_folders:
    folder_path = os.path.join(test_data_dir, emotion)
    
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (48, 48)) 
            test_images.append(img)
            test_labels.append(emotion_labels[emotion])  

test_images = np.array(test_images).reshape(-1, 48, 48, 1) / 255.0  # Normalize pixel values
test_labels = np.array(test_labels)


# %%
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")


# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predicting the labels for the test set
predicted_labels = np.argmax(model.predict(test_images), axis=1)

# Generating the confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Plotting the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_map.values(), yticklabels=emotion_map.values())
plt.title("Confusion Matrix")
plt.ylabel('Actual Emotion')
plt.xlabel('Predicted Emotion')
plt.show()


# %%
from sklearn.metrics import classification_report

emotion_map = {0: 'Angry', 1: 'Happy', 2: 'Sad'}
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
print(classification_report(test_labels, predicted_labels, target_names=list(emotion_map.values())))


# %%
model.fit(X_train, y_train, epochs=25, batch_size=64)


# %%
# def predict_emotion(image_path):
#     model = tf.keras.models.load_model("emotion_detection_model.h5")
    
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (48, 48))  
#     img = img.reshape(1, 48, 48, 1) / 255.0  

#     prediction = model.predict(img)
#     emotion = np.argmax(prediction)  

#     emotion_map = {0: 'Angry', 1: 'happy', 2: 'sad'}
#     print(f'Detected Emotion: {emotion_map[emotion]}')

#     if emotion_map[emotion] == 'sad':
#         print("Focusing on the Present: Quotes like “Some days are just bad days, that’s all. You have to experience sadness to know happiness, and I remind myself that not every day is going to be a good day, that’s just the way it is!” and “You are not alone. Everyone is fighting a battle you know nothing about” remind us to focus on the present moment and acknowledge that everyone experiences ups and downs. This can help us cultivate gratitude and appreciation for the present, leading to increased happiness.")
#     elif emotion_map[emotion] == 'Angry':
#         print('“I told my dentist I was feeling sad, and he said, ‘Well, you’re due for a filling… of happiness!”')
#     else:
#         print("Keep smiling")
    
# # Example usage:
# predict_emotion(r"C:\Users\Asus\OneDrive\Documents\Ip\Main\Image.png")


# %%
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("emotion_detection_model.h5")

# Define emotion map
emotion_map = {0: 'Angry', 1: 'Happy', 2: 'Sad'}

# Function to predict emotion from an image (face)
def predict_emotion_from_frame(img):
    img = cv2.resize(img, (48, 48))  
    img = img.reshape(1, 48, 48, 1) / 255.0  # Normalize

    prediction = model.predict(img)
    emotion = np.argmax(prediction)
    return emotion_map[emotion]

# Capture video from the default webcam (usually the built-in camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the video feed
    ret, frame = cap.read()

    if not ret:
        break  # If the frame is not captured properly, break out of the loop

    # Convert the captured frame to grayscale (necessary for emotion detection)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's face detector to locate faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) corresponding to the face
        face_roi = gray_frame[y:y + h, x:x + w]

        # Predict emotion for the face ROI
        emotion = predict_emotion_from_frame(face_roi)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the detected emotion on the frame above the face
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame with the emotion label
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to exit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# %%
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")


# %%
# import cv2
# image = cv2.imread(r"C:\Users\Asus\Pictures\Camera Roll\WIN_20241012_17_20_29_Pro.jpg")
# cv2.imshow("Nope",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# %%
import cv2
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("The camera is not opened")

else:
    result , image = cam.read()
    if result:
        filename = input("Write the name of the file")
        cv2.imwrite(f"the image is saved {filename}.jpg",image)
        print(f"the image is saved {filename}.jpg")
        cv2.imshow("Captured Image",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("The Camera is not Working")
        
cam.release()

        

# %%


# %%


# %%



