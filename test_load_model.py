import tensorflow as tf
model= tf.keras.models.load_model("model.h5")
print("Model loaded successfully!")
model.summary()
