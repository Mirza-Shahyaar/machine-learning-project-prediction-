import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Read the dataset
music_data = pd.read_csv('music.csv')

# Separate features and target
x = music_data.drop(columns=['genre'])
y = music_data['genre']

# Create and train the model
model = DecisionTreeClassifier()
model.fit(x, y)

# New data for prediction with the same feature names
new_data = pd.DataFrame([[21, 1], [22, 0]], columns=x.columns)

# Make predictions
predictions = model.predict(new_data)

# Print predictions
print(predictions)

