# PLANT DISEASE DETECTION MODEL (HIGH ACCURACY VERSION)
# This script uses transfer learning with a pre-trained MobileNetV2 model to achieve high accuracy
# in identifying plant diseases. It is optimized to run in a standard Google Colab environment.

# --- 1. Setup and Imports ---
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt

# --- GPU Check ---
def check_for_gpu():
    """Checks for a GPU and prints a warning if not found."""
    print("--- Checking for GPU ---")
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("WARNING: No GPU detected. Training will be very slow.")
        print("In Google Colab, go to 'Runtime' -> 'Change runtime type' and select 'GPU' as the hardware accelerator.")
    else:
        try:
            gpu_name = tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'Unknown')
            print(f"SUCCESS: GPU detected: {gpu_name}")
        except:
            print("SUCCESS: GPU detected.")
    print("----------------------\n")

# Colab helper for file uploads
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# --- 2. Data Loading and Preparation (using PlantVillage dataset) ---

def load_and_prepare_dataset():
    """
    Loads the 'plant_village' dataset, which contains images of healthy and diseased plants,
    and prepares it for the MobileNetV2 model.
    """
    print("Loading 'plant_village' dataset from TensorFlow Datasets...")

    # Split the data: 70% for training, 15% for validation, 15% for testing.
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'plant_village',
        split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    num_classes = ds_info.features['label'].num_classes
    class_names = ds_info.features['label'].names
    print(f"\nDataset loaded successfully. Found {num_classes} classes.")
    print("Example classes:", class_names[:5])


    # --- Data Preprocessing and Augmentation for Transfer Learning ---
    IMG_SIZE = 224 # MobileNetV2 is optimized for 224x224 images
    BATCH_SIZE = 32

    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    def preprocess_image(image, label):
        """Resizes images for the model."""
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE

    # Create efficient data pipelines
    ds_train = ds_train.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.shuffle(1000)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)

    ds_val = ds_val.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.batch(BATCH_SIZE)
    ds_val = ds_val.prefetch(buffer_size=AUTOTUNE)

    ds_test = ds_test.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE)
    ds_test = ds_test.prefetch(buffer_size=AUTOTUNE)

    return ds_train, ds_val, ds_test, num_classes, class_names

# --- 3. Build High-Accuracy Model with Transfer Learning ---

def build_transfer_learning_model(num_classes):
    """
    Builds a high-accuracy model using MobileNetV2 as a pre-trained base.
    """
    # Define the input shape
    inputs = layers.Input(shape=(224, 224, 3))

    # Pre-processing layer for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Load MobileNetV2 pre-trained on ImageNet, without its top classification layer
    base_model = MobileNetV2(input_shape=(224, 224, 3),
                             include_top=False,
                             weights='imagenet')

    # Freeze the base model to prevent its weights from being updated during initial training
    base_model.trainable = False

    # Pass the preprocessed input through the base model
    x = base_model(x, training=False)

    # Add a new classification head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x) # Regularization
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = tf.keras.Model(inputs, outputs)

    # Compile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

# --- 4. Train the Model ---

def train_model(model, ds_train, ds_val, epochs=25):
    """
    Trains the model with a callback for early stopping.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print("\n--- Starting Model Training ---")
    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_val,
        callbacks=[early_stopping]
    )
    print("\n--- Training Completed ---")
    return history

# --- 5. Plot Training History ---

def plot_training_history(history):
    """Plots the model's accuracy and loss over the training epochs."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

# --- 6. Prediction on User Image ---

def predict_user_image(model, class_names):
    """
    Prompts the user to upload an image and predicts its class.
    """
    print("\n--- Ready for Prediction ---")

    if IN_COLAB:
        print("Please upload an image of a plant leaf.")
        uploaded = files.upload()
        if not uploaded:
            print("No file uploaded. Skipping prediction.")
            return
        for fn in uploaded.keys():
            path = fn
            break
    else:
        path = input("Enter the path to your image file: ")

    if not os.path.exists(path):
        print("File not found. Please provide a valid path.")
        return

    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class.replace('___', ' ')}\nConfidence: {confidence:.2f}%")
    plt.axis("off")
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    check_for_gpu()

    # Step 1: Load the plant disease dataset
    ds_train, ds_val, ds_test, num_classes, class_names = load_and_prepare_dataset()

    # Step 2: Build the high-accuracy transfer learning model
    model = build_transfer_learning_model(num_classes)

    # Step 3: Train the model
    history = train_model(model, ds_train, ds_val, epochs=25)

    # Step 4: Evaluate the model's final accuracy on unseen test data
    print("\n--- Evaluating Final Model Accuracy ---")
    loss, accuracy = model.evaluate(ds_test)
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

    # Step 5: Plot results
    plot_training_history(history)

    # Step 6: Save the model
    model.save("plant_disease_detector_high_accuracy.h5")
    print("\nModel saved as 'plant_disease_detector_high_accuracy.h5'")

    # Step 7: Allow user to predict an image
    predict_user_image(model, class_names)




