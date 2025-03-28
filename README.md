# Movie-Recommendation-System-PyTorch

This project focuses on developing a deep learning–based movie recommendation system
using the MovieLens dataset. The goal is to predict how users would rate movies based on their
historical preferences and enriched movie metadata.

The dataset used is the MovieLens 100K, which contains:

• 100,836 ratings from 610 users on 9,742 movies
• Each user rated at least 20 movies
• Additional information was extracted using the links.csv file to connect with IMDb
  and TMDb APIs, enriching the dataset with fields such as:
    • Budget
    • Revenue
    • Popularity
    • Average TMDb rating
    • Movie metadata was further processed to extract genres (as multi-label inputs) and release
      decades (used as categorical embeddings)


## Overview of preprocessing the dataset

• Linked each movie to external databases (just TMDb) to extract additional features such as
budget, revenue, popularity, and average rating.

• Handled missing values and ensured consistent data types across features.

• Extracted and encoded genres as multi-label categorical features using embeddings.

• Calculated release decade and encoded it as a separate categorical feature.

• Normalized numerical features (e.g., budget, revenue) and handled skewness using
transformation techniques.

• Introduced time-decay weighted ratings, giving more importance to recent user feedback.


## Model architecture

  To capture the complex relationships between users, movies, and metadata, a Neural
Collaborative Filtering (NCF) model was implemented using PyTorch. The architecture wasdesigned to handle both categorical and numerical inputs, enabling the system to learn rich, nonlinear patterns in the data.

  Neural Collaborative Filtering (NCF) is a deep learning–based approach to recommendation systems that extends traditional collaborative filtering by replacing the dot-product interaction (used in matrix factorization) with a learnable neural network. NCF is capable of learning nonlinear and more complex patterns in user-item interactions by leveraging the power of deep neural networks.
  
## Model Inputs

The model accepts the following inputs:

• User ID: encoded via an embedding layer

• Movie ID: encoded via an embedding layer

• Genres: multi-label categorical input encoded via embeddings and averaged

• Decade: categorical input encoded via an embedding

• Numerical Features: (budget, revenue, etc.) passed through a linear layer

  Each of these input types is projected into a shared embedding space and then concatenated into a
single vector representing the user-movie interaction.

## Network Architecture

The combined embedding vector is passed through a sequence of fully connected layers:

• Input layer: concatenated user, movie, genre, decade, and numerical embeddings

• Hidden layers: two dense layers with ReLU activation

• Output layer: a single neuron that predicts the expected movie rating

## Results obtained

  Each training run was performed over 30 epochs using the Adam optimizer, with mini-
batch gradient descent (batch size of 64). After training, models were evaluated using multiple
performance metrics to compare the impact of each loss function on prediction accuracy.To evaluate the performance of the recommendation model, the system was trained using three different loss functions: MSELoss, L1Loss, and SmoothL1Loss. Each model was trained on time-decay weighted ratings and evaluated using a set of standard regression metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² Score.


## Ways to improve outcome

  One potential enhancement is to incorporate more contextual information about the movies, such as
director, actors, or keywords, which can be extracted from external sources like TMDb.

  On the modeling side, experimenting with more expressive architectures—such as attention
mechanisms or Transformer-based models—could allow the system to capture more subtle patterns
in user-item interactions.

  Techniques like dropout, batch normalization, or learning rate scheduling might improve
generalization and prevent overfitting. Also, playing more with the hyperparameters might have
improved the final result.
