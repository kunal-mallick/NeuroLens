"""
Clinical Model - Lightweight 2D U-Net for fast segmentation.
TensorFlow / Keras implementation.
Builds the model and saves it to disk with logging.
"""

import os
import logging
import tensorflow as tf
from tensorflow.keras import layers, models

# -------- CONFIG --------
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1
MODEL_PATH = "models/clinical_unet_2d.h5"

# -------- LOGGING SETUP --------
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "model_build.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------- MODEL DEFINITION --------
def conv_block(x, filters):
    x = layers.Conv2D(filters, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPooling2D((2,2))(f)
    return f, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)

    b1 = conv_block(p3, 256)

    d1 = decoder_block(b1, s3, 128)
    d2 = decoder_block(d1, s2, 64)
    d3 = decoder_block(d2, s1, 32)

    outputs = layers.Conv2D(1, (1,1), activation="sigmoid")(d3)
    model = models.Model(inputs, outputs, name="Clinical_UNet_2D")
    return model

# -------- MAIN SCRIPT --------
if __name__ == "__main__":
    logging.info("Starting model build process...")
    os.makedirs("models", exist_ok=True)

    model = build_unet()
    model.summary(print_fn=lambda x: logging.info(x))
    
    model.save(MODEL_PATH)
    logging.info(f"✅ Model saved successfully at {MODEL_PATH}")
    print(f"Model built and saved at {MODEL_PATH}")