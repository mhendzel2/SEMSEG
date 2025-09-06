"""
U-Net model for segmentation.
"""
import tensorflow as tf
from tensorflow.keras import layers

def unet_model(input_size=(256, 256, 1), num_classes=2):
    """
    Create a 2D U-Net model.
    """
    inputs = tf.keras.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c5 = layers.Dropout(0.2)(c5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.1)(c7)
    c7 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c7)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return model

def predict_slices(model, data):
    """
    Predict segmentation slice by slice for a 3D volume.
    Assumes data is (slices, height, width).
    """
    import numpy as np

    slices, height, width = data.shape

    # Pre-allocate prediction array
    if model.output_shape[-1] > 1:
        # Multi-class segmentation
        predictions = np.zeros((slices, height, width, model.output_shape[-1]), dtype=np.float32)
    else:
        # Binary segmentation
        predictions = np.zeros((slices, height, width), dtype=np.float32)

    for i in range(slices):
        slice_data = data[i, :, :]

        # Normalize and add channel dimension
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
        slice_data = np.expand_dims(slice_data, axis=-1)

        # Resize to model input size if necessary
        if slice_data.shape[:2] != model.input_shape[1:3]:
            original_shape = slice_data.shape[:2]
            slice_data = tf.image.resize(slice_data, model.input_shape[1:3])
        else:
            original_shape = None

        # Predict
        pred_slice = model.predict(np.expand_dims(slice_data, axis=0))[0]

        # Resize back to original shape
        if original_shape:
            pred_slice = tf.image.resize(pred_slice, original_shape)

        if model.output_shape[-1] > 1:
            predictions[i, :, :, :] = pred_slice
        else:
            predictions[i, :, :] = pred_slice[:, :, 0]

    return predictions
