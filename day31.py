# Cat vs Dog Prediction using CNN (TensorFlow / Keras)
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# -----------------------------
# Load Dataset
# -----------------------------
X_train = np.loadtxt('input.csv', delimiter=',')
Y_train = np.loadtxt('labels.csv', delimiter=',')

X_test = np.loadtxt('input_test.csv', delimiter=',')
Y_test = np.loadtxt('labels_test.csv', delimiter=',')

# -----------------------------
# Reshape Data
# -----------------------------
X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)

X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

# -----------------------------
# Normalize Data
# -----------------------------
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Training data shape:", X_train.shape, Y_train.shape)
print("Testing data shape:", X_test.shape, Y_test.shape)

# -----------------------------
# Display Random Training Image
# -----------------------------
idx = random.randint(0, len(X_train) - 1)
plt.imshow(X_train[idx])
plt.title("Random Training Image")
plt.axis('off')
plt.show()

# -----------------------------
# Build CNN Model
# -----------------------------
model = Sequential()

model.add(Conv2D(23, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(23, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(
    X_train,
    Y_train,
    epochs=5,
    batch_size=64
)

# -----------------------------
# Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# -----------------------------
# Make Prediction
# -----------------------------
idx2 = random.randint(0, len(X_test) - 1)
plt.imshow(X_test[idx2])
plt.title("Test Image")
plt.axis('off')
plt.show()

y_pred = model.predict(X_test[idx2].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5

if y_pred == 0:
    result = "dog"
else:
    result = "cat"

print("Our Model says it is a:", result)