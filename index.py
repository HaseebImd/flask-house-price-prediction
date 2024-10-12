from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)


# Enable CORS for all routes and all origins
CORS(app)

# Load the trained model (make sure the path is correct)
model = joblib.load("house_price_model.pkl")


# Define the route for house price prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the data from the request
        data = request.json

        # Validate all required fields are present
        required_fields = [
            "bathrooms",
            "square_footage",
            "lot_size",
            "year_built",
            "bedrooms",
        ]
        for field in required_fields:
            if field not in data or data[field] is None:
                return (
                    jsonify({"error": f"'{field}' is required"}),
                    400,
                )  # Return an error with status 400

        # Extract features from the request (these must match the training features)
        # Convert the fields to numeric values (float or int)
        features = [
            float(data["bathrooms"]),
            float(data["square_footage"]),
            float(data["lot_size"]),
            int(data["year_built"]),
            int(data["bedrooms"]),
        ]

        # Convert the features into the required format (e.g., a 2D array for prediction)
        input_data = np.array([features])

        # Make the prediction
        prediction = model.predict(input_data)

        # Round the predicted house price to two decimal places
        predicted_price = round(prediction[0], 2)

        # Return the predicted house price as JSON
        return jsonify({"predicted_price": predicted_price})

    except Exception as e:
        # Handle any errors during prediction
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to port 5000 if PORT isn't set
    app.run(host="0.0.0.0", port=port)
