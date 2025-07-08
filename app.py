from flask import Flask, request, render_template
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load('titanic_model.pkl')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction_text='Survived' if prediction == 1 else
'Not Survived')
if __name__ == '__main__':
    app.run(debug=True)