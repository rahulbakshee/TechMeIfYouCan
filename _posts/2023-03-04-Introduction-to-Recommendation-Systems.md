# Recommendation Syetms

![social media]({{ '/images/2023-03-04-social.jpg' | relative_url }}){: style="width: 600px; max-width: 100%;"}

Recommendation systems are a type of machine learning technology that have become increasingly popular in recent years, especially with the rise of e-commerce, social media, and entertainment platforms. These systems are designed to suggest products, services, or content to users based on their past behavior, preferences, and other relevant data.

There are two main types of recommendation systems: content-based and collaborative filtering.

Content-based recommendation systems analyze the characteristics of the items a user has interacted with in the past and recommend similar items. For example, if a user has watched several romantic comedies, a content-based system might recommend more romantic comedies or movies with similar themes or actors. These systems use machine learning algorithms to extract features from the items and build user profiles based on their preferences.

Collaborative filtering, on the other hand, recommends items based on the preferences of similar users. In this approach, the system analyzes the behavior and preferences of a large number of users and identifies groups of users who share similar tastes. If a user belongs to a group that likes a certain product, the system might recommend that product to them. Collaborative filtering algorithms can be further classified into two subcategories: user-based and item-based.

User-based collaborative filtering compares the preferences of a user to those of other users and recommends items that similar users have liked in the past. Item-based collaborative filtering, on the other hand, analyzes the similarities between items and recommends items that are similar to those that the user has already interacted with.

In order to build a recommendation system, a large amount of data is required. This data can include user behavior (e.g. purchase history, clicks, ratings), item attributes (e.g. genre, price, release date), and user demographics (e.g. age, location, gender). The system then uses this data to train machine learning models that can make personalized recommendations.

One common algorithm used in recommendation systems is the matrix factorization method. This method decomposes the user-item interaction matrix into two lower dimensional matrices, one for users and one for items, that capture the latent features that explain the observed interactions. These latent features might represent characteristics such as genre, mood, or style, that are not explicitly defined in the input data.

Another important aspect of recommendation systems is evaluation. Since these systems are often used to make important decisions that can impact user experience and revenue, it is crucial to measure their effectiveness. Common evaluation metrics for recommendation systems include accuracy, coverage, and diversity.

In conclusion, recommendation systems are a powerful tool for personalizing user experiences and driving engagement and revenue in various industries. These systems use machine learning algorithms to analyze user behavior and preferences and make personalized recommendations. By leveraging large amounts of data and advanced techniques such as matrix factorization, recommendation systems can provide valuable insights into user preferences and help businesses optimize their offerings.


> Let us dive into code:
 We are going to use Tensorflow recommenders libarary and MovieLens dataset for handson

# TensorFlow Recommenders 
TensorFlow Recommenders is a library for building recommender system models using TensorFlow.

It helps with the full workflow of building a recommender system: data preparation, model formulation, training, evaluation, and deployment.

It's built on Keras and aims to have a gentle learning curve while still giving you the flexibility to build complex models.


### installs
```
!pip install -q tensorflow-recommenders
!pip install -q --upgrade tensorflow-datasets
```

### imports
```
from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
```

### data loading
```
# Ratings data.
ratings = tfds.load('movielens/100k-ratings', split="train")
# Features of all the available movies.
movies = tfds.load('movielens/100k-movies', split="train")

# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})
movies = movies.map(lambda x: x["movie_title"])
```

### data preprocessing
```
# Build vocabularies to convert user ids and movie titles into integer indices for embedding layers:
user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))

movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(movies)
```

### model definition
```
# define a TFRS model by inheriting from tfrs.Model and implementing the compute_loss method:

class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(
      self,
      user_model: tf.keras.Model,
      movie_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.movie_model = movie_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.movie_model(features["movie_title"])

    return self.task(user_embeddings, movie_embeddings)

```

### model creation
```
# Define the two models and the retrieval task.
# Define user and movie models.

user_model = tf.keras.Sequential([
    user_ids_vocabulary,
    tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
])
movie_model = tf.keras.Sequential([
    movie_titles_vocabulary,
    tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
])

# Define your objectives.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    movies.batch(128).map(movie_model)
  )
)
```

### model training and evaluation
```
# Create a retrieval model.
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train for 3 epochs.
model.fit(ratings.batch(4096), epochs=3)

# Use brute-force search to set up retrieval using the trained representations.
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    movies.batch(100).map(lambda title: (title, model.movie_model(title))))

# Get some recommendations.
_, titles = index(np.array(["42"]))
print(f"Top 3 recommendations for user 42: {titles[0, :3]}")

```