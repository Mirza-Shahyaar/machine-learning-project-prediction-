# Music Genre Prediction

This repository contains a series of Python scripts for training, evaluating, and using a Decision Tree Classifier model to predict music genres. The repository also includes functionality for visualizing the decision tree and saving/loading models for predictions.

## Description

### 1. **Training and Visualizing the Decision Tree**

The `train_and_visualize.py` script trains a Decision Tree Classifier model on a dataset and exports the decision tree to a `.dot` file for visualization. The `.dot` file can be used with Graphviz to generate a graphical representation of the tree structure.

**Usage:**

```sh
python train_and_visualize.py
```

**Features:**
- Loads the dataset `music.csv`
- Trains the Decision Tree Classifier
- Exports the decision tree to `music-recommender.dot`

### 2. **Making Predictions with a Trained Model**

The `predict_music_genre.py` script demonstrates how to train a Decision Tree Classifier and use it to make predictions on new data. This script ensures that the new data is formatted correctly to match the training data's features.

**Usage:**

```sh
python predict_music_genre.py
```

**Features:**
- Trains the model on the dataset
- Prepares new data for prediction
- Prints the predictions for the new data

### 3. **Evaluating Model Accuracy**

The `evaluate_model.py` script trains a Decision Tree Classifier model, evaluates its performance on a test set, and prints the accuracy score. The dataset is split into training and testing sets to assess the model's performance.

**Usage:**

```sh
python evaluate_model.py
```

**Features:**
- Splits the dataset into training and testing sets
- Trains the model on the training set
- Evaluates and prints the accuracy on the test set

### 4. **Loading a Pre-trained Model and Making Predictions**

The `load_and_predict.py` script demonstrates how to load a pre-trained Decision Tree model and make predictions on new data. This script assumes the model has been saved using `joblib` and that the feature names used during training are known.

**Usage:**

```sh
python load_and_predict.py
```

**Features:**
- Loads the pre-trained model from `music-recommender.joblib`
- Prepares new data with the correct feature names
- Prints the predictions for the new data

### 5. **Saving a Trained Model**

The `train_and_save_model.py` script trains a Decision Tree Classifier model and saves it to a file using `joblib`. This allows the model to be reused later without retraining.

**Usage:**

```sh
python train_and_save_model.py
```

**Features:**
- Trains the model on the dataset
- Saves the trained model to `music-recommender.joblib`

## Prerequisites

- Python 3.x
- `pandas`
- `scikit-learn`
- `joblib`
- `graphviz` (for tree visualization)

You can install the required packages using `pip`:

```sh
pip install pandas scikit-learn joblib graphviz
```

**Note:** Ensure Graphviz is installed on your system. You can download it from [Graphviz's official website](https://graphviz.gitlab.io/download/).

## Accuracy Measurement

To measure the accuracy of the model, use the `evaluate_model.py` script, which calculates the accuracy score of the model on a test dataset.

## Tree Visualization

To visualize the decision tree, use the `train_and_visualize.py` script to generate a `.dot` file. Convert the `.dot` file to an image format using Graphviz tools:

```sh
dot -Tpng music-recommender.dot -o music-recommender.png
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
