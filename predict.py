import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from class_names import class_names

# Load trained model
model = load_model("models/skin_disease_model.keras")

# Image path (change this later)
img_path = "test_image.jpg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("\n===== Prediction Result =====")
print("Disease:", class_names[predicted_class])
print(f"Confidence: {confidence:.2f}%")

print("\nAll Probabilities:")

for i, disease in enumerate(class_names):
    prob = prediction[0][i] * 100
    print(f"{disease}: {prob:.2f}%")