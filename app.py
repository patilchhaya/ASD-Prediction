from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load(open("model.pkl", "rb"))

# Define the route for the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Define the route for predicting ASD
@app.route("/predict", methods=["POST"])
def predict_ASD():
    # Extract features from the form
    Age = int(request.form["Age"])
    Gender = int(request.form["Gender"])
    Ethnicity = int(request.form["Ethnicity"])
    Family_History = int(request.form["Family_History"])
    Developmental_Milestones = int(request.form["Developmental_Milestones"])
    Social_Interactions_Score = float(request.form["Social_Interactions_Score"])
    Communication_Score = float(request.form["Communication_Score"])
    Repetitive_Behaviors_Score = float(request.form["Repetitive_Behaviors_Score"])
    Hyperactivity_Score = float(request.form["Hyperactivity_Score"])

    # Make prediction
    features = np.array([[Age, Gender, Ethnicity, Family_History, Developmental_Milestones, 
                          Social_Interactions_Score, Communication_Score, 
                          Repetitive_Behaviors_Score, Hyperactivity_Score]])
    result = model.predict(features)

    # Process the prediction result
    if result == 1:
        prediction = "Person is suffering from ASD"
        color = "green"
    else:
        prediction = "Person is not suffering from ASD"
        color = "red"

    # Render the template with the prediction result
    return render_template("index.html", prediction=prediction, color=color)

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True, port=5001)
