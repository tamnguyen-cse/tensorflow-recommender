import pprint
from typing import Dict, Text

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

# Users data.
tracking_type = {'timestamp': str, 'user_id': str, 'store_name': str, 'affiliate': str, 'card_linked': str,
                 'e_gift': str, 'offer_code': str, 'referral_type': str, 'referral_message': str, }
tracking_data = pd.read_csv("tracking_data.csv", dtype=tracking_type)
# Features of all the available gifts.
gift_card_type = {'store_name': str, 'discount': str, 'category': str}
gift_card_data = pd.read_csv("gift_card.csv")

users = tf.data.Dataset.from_tensor_slices(dict(tracking_data))
products = tf.data.Dataset.from_tensor_slices(dict(gift_card_data))

for x in users.take(1).as_numpy_iterator():
    pprint.pprint(x)
for x in products.take(1).as_numpy_iterator():
    pprint.pprint(x)

users = users.map(lambda x: {
    "store_name": x["store_name"],
    "user_id": x["user_id"],
    "e_gift": x["e_gift"],
})
products = products.map(lambda x: x["store_name"])

tf.random.set_seed(42)
shuffled = users.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

product_names = products.batch(1_000)
user_ids = users.batch(1_000_000).map(lambda x: x["user_id"])

unique_product_names = np.unique(np.concatenate(list(product_names)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

pprint.pprint(unique_product_names[:10])

embedding_dimension = 32

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
    # We add an embedding to account for unknown tokens.
    tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

product_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=unique_product_names, mask_token=None),
    tf.keras.layers.Embedding(len(unique_product_names) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
    candidates=products.batch(128).map(product_model)
)

task = tfrs.tasks.Retrieval(
    metrics=metrics
)


class ProductModel(tfrs.Model):

    def __init__(self, user_model, product_model):
        super().__init__()
        self.product_model: tf.keras.Model = product_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the product features and pass them into the product model,
        # getting embeddings back.
        positive_product_embeddings = self.product_model(features["store_name"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_product_embeddings)


model = ProductModel(user_model, product_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8000).cache()
cached_test = test.batch(2000).cache()

model.fit(cached_train, epochs=5)
model.evaluate(cached_test, return_dict=True)

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k=20)
# recommends products out of the entire product dataset.
index.index_from_dataset(
    tf.data.Dataset.zip((products.batch(100), products.batch(100).map(model.product_model)))
)

# Get recommendations.
scores, brands = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {brands}")
print(f"Recommendation scores for user 42: {scores}")

# Export the query model.
path = "/tmp/gift_model"

# Save the index.
tf.saved_model.save(index, path)

# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load(path)
