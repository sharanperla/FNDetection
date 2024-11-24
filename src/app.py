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

# Configure Gemini AI API
GOOGLE_API_KEY ="AIzaSyAGsQQrde1QNNBVi1LpUuLIuIetClxmAwM"
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Configure the model
    model = genai.GenerativeModel('gemini-pro')
else:
    logger.warning("Google API key not found in environment variables")

def get_related_articles(text, is_fake):
    """Generate related factual articles using Gemini AI"""
    try:
        # Check if API key is configured
        if not GOOGLE_API_KEY:
            return "Related articles feature is unavailable (API key not configured)"

        prompt = f"""
        {'The following news might be fake' if is_fake else 'Regarding this news'}: "{text}"
        Please provide 2 factual, verified articles related to this topic.
        Include:
        1. Article title
        2. Brief summary (2-3 sentences)
        3. Publication date
        4. Source (if available)
        {'Focus on factual information and cite reliable sources.' if is_fake else ''}
        """

        # For testing without API key, return dummy data
        if not GOOGLE_API_KEY:
            return "API key not configured - Related articles unavailable"

        # Generate content using Gemini
        response = model.generate_content(prompt)
        
        # Check if the response is valid
        if response.text:
            return response.text
        else:
            return "No relevant articles found"

    except Exception as e:
        logger.error(f"Error in get_related_articles: {str(e)}")
        return str(e)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, bool)):
            return str(obj)
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

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

        # Preprocess text and make predictions
        try:
            text_tfidf = tfidf.transform([text])
            log_pred = int(log_model.predict(text_tfidf)[0])
            dt_pred = int(dt_model.predict(text_tfidf)[0])
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': 'Error processing text'}), 500

        # Generate result
        is_fake = str(log_pred == 0 or dt_pred == 0)
        related_articles = get_related_articles(text, is_fake == 'True')

        result = {
            'logistic_regression_prediction': 'Fake' if log_pred == 0 else 'Real',
            'decision_tree_prediction': 'Fake' if dt_pred == 0 else 'Real',
            'is_fake': is_fake,
            'related_articles': related_articles
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

# Error handlers
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    if isinstance(e, HTTPException):
        return jsonify({'error': str(e)}), e.code
    return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    # Check if critical components are missing
    if None in [log_model, dt_model, tfidf]:
        logger.warning("Some models failed to load. Application may not function correctly.")
    
    if not GOOGLE_API_KEY:
        logger.warning("Google API key not set. Related articles feature will be disabled.")
    
    app.run(debug=True)