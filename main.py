import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os

batch_size = 32
img_height = 224
img_width = 224


def classify_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class


datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

# dataset/pictures/ - папка с отсортированными изображениями
train_generator = datagen.flow_from_directory(
    'dataset/pictures/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')
validation_generator = datagen.flow_from_directory(
    'dataset/pictures/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

num_classes = len(train_generator.class_indices)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)

model.save('landscape_classifier.h5')
model = tf.keras.models.load_model('landscape_classifier.h5')

# Изображения для классификации должны лежать в этой папке!!!
data_folder = 'data/'

# Классы должны соответствовать папкам в директории dataset
classes = ['landscape', 'mountain', 'desert', 'sea', 'beach', 'island', 'japan']

for filename in os.listdir(data_folder):
    if filename.endswith('.jpg'):
        img_path = os.path.join(data_folder, filename)
        predicted_class = classify_image(img_path)
        print(f"Изображение '{filename}' принадлежит классу '{classes[predicted_class]}'.")

test_loss, test_accuracy = model.evaluate(validation_generator)
print("Точность на тестовом наборе данных:", test_accuracy)
print("Количество потерь в тестовом наборе данных:", test_loss)
