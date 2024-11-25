from flask import Flask, request, jsonify, render_template
import pickle
import google.generativeai as genai
import os
import json
from datetime import datetime
import logging
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize variables to None
log_model = None
dt_model = None
tfidf = None

# Load models with error handling
try:
    # Check if model files exist first
    model_files = ['log_model.pkl', 'dt_model.pkl', 'tfidf_vectorizer.pkl']
    for file in model_files:
        if not os.path.exists(file):
            logger.error(f"Model file not found: {file}")
            raise FileNotFoundError(f"Required model file {file} is missing")

    # Load models
    with open('log_model.pkl', 'rb') as log_file:
        log_model = pickle.load(log_file)

    with open('dt_model.pkl', 'rb') as dt_file:
        dt_model = pickle.load(dt_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
        tfidf = pickle.load(vec_file)

except Exception as e:
    logger.error(f"Error during model loading: {str(e)}")

# Configure Gemini AI
GOOGLE_API_KEY ="AIzaSyAGsQQrde1QNNBVi1LpUuLIuIetClxmAwM"
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Configure the model
    model = genai.GenerativeModel('gemini-pro')
else:
    logger.warning("Google API key not found in environment variables")

def get_gemini_prediction(text):
    """Get prediction from Gemini AI"""
    try:
        if not GOOGLE_API_KEY:
            return None, "API key not configured"

        prompt = f"""
        Analyze if the following news is fake or real. Only respond with a single word: 'Fake' or 'Real'.
        News: "{text}"
        """

        response = model.generate_content(prompt)
        prediction = response.text.strip().lower()
        
        # Normalize the response to match our format
        if 'fake' in prediction:
            return 'Fake', None
        elif 'real' in prediction:
            return 'Real', None
        else:
            return None, "Unclear prediction from Gemini"

    except Exception as e:
        logger.error(f"Error in Gemini prediction: {str(e)}")
        return None, str(e)

def get_explanation(text, gemini_prediction, log_pred, dt_pred):
    """Generate detailed explanation using Gemini AI"""
    try:
        if not GOOGLE_API_KEY:
            return "Explanation feature is unavailable (API key not configured)"

        prompt = f"""
        Analyze this news: "{text}"
        
        Our analysis shows:
        - AI Prediction: {gemini_prediction}
        - Logistic Regression Model: {"Fake" if log_pred == 0 else "Real"}
        - Decision Tree Model: {"Fake" if dt_pred == 0 else "Real"}

        Please provide:
        1. A detailed explanation of why this news might be {gemini_prediction} (2-3 paragraphs)
        2. Key indicators that led to this conclusion
        3. Two relevant, factual articles about this topic:
           - Article title
           - Brief summary
           - Publication date
           - Source

        Format the response with clear headings and bullet points.
        """

        response = model.generate_content(prompt)
        return response.text if response.text else "No explanation available"

    except Exception as e:
        logger.error(f"Error in get_explanation: {str(e)}")
        return str(e)

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return jsonify({'error': 'Error loading page'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if models are loaded
        if None in [log_model, dt_model, tfidf]:
            return jsonify({
                'error': 'Models not properly loaded. Please check server logs.'
            }), 500

        # Get and validate input
        if not request.is_json:
            return jsonify({'error': 'Invalid request format. Expected JSON'}), 400

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400

        # Get ML model predictions
        text_tfidf = tfidf.transform([text])
        log_pred = int(log_model.predict(text_tfidf)[0])
        dt_pred = int(dt_model.predict(text_tfidf)[0])

        # Get Gemini prediction
        gemini_prediction, error = get_gemini_prediction(text)
        if error:
            return jsonify({'error': f'Error with Gemini prediction: {error}'}), 500

        # Compare predictions
        ml_fake = log_pred == 0 or dt_pred == 0
        gemini_fake = gemini_prediction == 'Fake'

        # If predictions differ, use Gemini's prediction
        if ml_fake != gemini_fake:
            log_pred = 0 if gemini_fake else 1
            dt_pred = 0 if gemini_fake else 1

        # Get detailed explanation
        explanation = get_explanation(text, gemini_prediction, log_pred, dt_pred)

        result = {
            'logistic_regression_prediction': 'Fake' if log_pred == 0 else 'Real',
            'decision_tree_prediction': 'Fake' if dt_pred == 0 else 'Real',
            'gemini_prediction': gemini_prediction,
            'is_fake': str(gemini_fake),
            'explanation': explanation
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    if not GOOGLE_API_KEY:
        logger.warning("Google API key not set. Gemini features will be disabled.")
    
    app.run(debug=True)