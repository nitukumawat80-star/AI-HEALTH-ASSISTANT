from flask import Flask, render_template, request
import pickle

# Flask app create
app = Flask(__name__)

# Trained model load
model = pickle.load(open("model/disease_model.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    # Form se symptoms lena
    symptoms = [int(x) for x in request.form.values()]

    # Prediction
    prediction = model.predict([symptoms])

    # Result show
    return "Predicted Disease: " + prediction[0]

# Run server
if __name__ == "__main__":
    app.run(debug=True)