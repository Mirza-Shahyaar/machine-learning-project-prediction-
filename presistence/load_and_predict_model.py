import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the pre-trained model
model = joblib.load('music-recommender.joblib')

# New data for prediction with the correct feature names
# Assuming the original features were 'age' and 'gender'
new_data = pd.DataFrame([[21, 1], [22, 0]], columns=['age', 'gender'])

# Make predictions
predictions = model.predict(new_data)
print(predictions)
