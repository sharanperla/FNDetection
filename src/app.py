from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained models and vectorizer from the src folder
with open('log_model.pkl', 'rb') as log_file:
    log_model = pickle.load(log_file)

with open('dt_model.pkl', 'rb') as dt_file:
    dt_model = pickle.load(dt_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    tfidf = pickle.load(vec_file)

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route for fake news detection
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    # Preprocess text and vectorize
    text_tfidf = tfidf.transform([text])
    
    # Get predictions from both models
    log_pred = log_model.predict(text_tfidf)[0]
    dt_pred = dt_model.predict(text_tfidf)[0]
    
    result = {
        'logistic_regression_prediction': 'Real' if log_pred == 1 else 'Fake',
        'decision_tree_prediction': 'Real' if dt_pred == 1 else 'Fake'
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
