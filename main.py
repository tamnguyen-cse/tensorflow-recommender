import tensorflow as tf

path = "/tmp/gift_model"

# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load(path)

# Pass a user id in, get top predicted product titles back.
scores, titles = loaded(["1"])

print(f"Recommendation titles: {titles}")
print(f"Recommendation scores: {scores}")
