# Fashion MNIST class labels
# 0 → T-shirt/top
# 1 → Trouser
# 2 → Pullover
# 3 → Dress
# 4 → Coat
# 5 → Sandal
# 6 → Shirt
# 7 → Sneaker
# 8 → Bag
# 9 → Ankle boot


import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


# Load Fashion MNIST dataset
# This dataset contains 28x28 grayscale images of clothing items
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# Normalize pixel values to range [0, 1]
# This helps the model train faster and more stably
x_train = x_train / 255.0
x_test = x_test / 255.0


# CNN expects input in (height, width, channels) format
# Since images are grayscale, channel = 1
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Reshaped training data:", x_train.shape)
print("Reshaped testing data:", x_test.shape)


# Build the CNN model
# Using two convolution layers followed by fully connected layers
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout to reduce overfitting
    layers.Dense(10, activation='softmax')  # 10 output classes
])


# Compile the model
# Using Adam optimizer and sparse categorical crossentropy loss
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Train the model
# Validation data is used to monitor performance on unseen data
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)


# Evaluate model performance on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test Accuracy:", test_accuracy)


# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Display predictions for first 20 test images
for i in range(20):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()
    
    prediction = model.predict(np.expand_dims(x_test[i], axis=0))
    print("Predicted label:", np.argmax(prediction))
    print("Actual label:", y_test[i])
    print("-" * 30)
