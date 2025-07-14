import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
#Load the dataset
df =pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
#Simple preprocessing
df = df[["Sex","Fare","Survived"]].dropna()
df['Sex'] = df["Sex"].map({'male':0,'female':1})
X = df[['Sex','Fare']]
y = df['Survived']
#Train the model and save
model = LogisticRegression()
model.fit(X,y)
joblib.dump(model,'titanic_model.pkl')