import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
# Load dataset
df =pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
# Drop missing and unnecessary columns
df.drop(['Name', 'Cabin', 'Ticket', 'Embarked'], axis=1, inplace=True)
df.dropna(inplace=True)
# Convert categorical column
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
# Define features and target
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']
# Train the model

model = LogisticRegression()
model.fit(X, y)
# Save model
joblib.dump(model, 'titanic_model.pkl')