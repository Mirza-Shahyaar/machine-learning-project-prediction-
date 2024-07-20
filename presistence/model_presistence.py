import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset into a DataFrame
music_data = pd.read_csv('music.csv')

# Separate features and target variable
x = music_data.drop(columns=['genre'])  # Features
y = music_data['genre']  # Target

# Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(x, y)

# Save the trained model to a file using joblib
joblib.dump(model, 'music-recommender.joblib')

# predictions=model.predict([[21,1],[22,0]])
# print(predictions)
