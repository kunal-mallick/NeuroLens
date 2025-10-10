"""
Research Model - 3D Attention U-Net (build & save only)
TensorFlow / Keras implementation with logging.
"""

import os
import logging
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------- CONFIG ----------
IMG_DEPTH = 128
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 4
MODEL_PATH = "models/research_attention_unet_3d.h5"

# ---------- LOGGING ----------
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "build_model.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- MODEL COMPONENTS ----------
def conv_block(x, filters):
    x = layers.Conv3D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv3D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPooling3D((2, 2, 2))(f)
    return f, p

def decoder_block(x, skip, filters):
    # Standard decoder block without attention to avoid shape mismatch
    up = layers.Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding="same")(x)
    concat = layers.Concatenate()([up, skip])
    return conv_block(concat, filters)

def build_unet_3d(input_shape=(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    # Bottleneck
    b = conv_block(p4, 512)

    # Decoder
    d1 = decoder_block(b, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    outputs = layers.Conv3D(1, 1, activation="sigmoid")(d4)
    model = models.Model(inputs, outputs, name="Research_UNet3D")
    
    return model

# ---------- MAIN ----------
if __name__ == "__main__":
    logger.info("🔹 Building 3D U-Net model...")
    os.makedirs("models", exist_ok=True)
    
    model = build_unet_3d()
    model.summary(print_fn=lambda x: logger.info(x))
    
    model.save(MODEL_PATH)
    logger.info(f"✅ Model saved to {MODEL_PATH}")