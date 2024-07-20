import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset into a DataFrame
music_data = pd.read_csv('music.csv')

# Separate features and target variable
x = music_data.drop(columns=['genre'])  # Features
y = music_data['genre']  # Target

# Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
predictions = model.predict(x_test)

# Evaluate the model's performance using accuracy
score = accuracy_score(y_test, predictions)
print(score)
