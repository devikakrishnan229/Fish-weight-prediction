from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)

# Load your data
df = pd.read_csv('Fish.csv')

# Preprocess your data (if needed)
# Example: Drop columns and split data
X = df.drop('Species', axis=1).values
y = df['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train your model
model = RandomForestClassifier(random_state=100)
model.fit(X_train_scaled, y_train)

# Render the HTML template for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle POST request for predictions
@app.route('/predict', methods=['POST'])
def predict():
    weight = float(request.form['weight'])
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])
      

    # Prepare input for prediction
    input_data = np.array([[weight, length1, length2, length3, height, width]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Get prediction
    prediction = model.predict(input_data_scaled)[0]

    # Render prediction result in HTML template
    return render_template('index.html', prediction_text=f'The predicted species is {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
