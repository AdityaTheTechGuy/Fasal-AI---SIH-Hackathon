import tempfile
import os
import pickle
import numpy as np
import io
from flask import Flask, request, render_template, jsonify, session, send_file, redirect, url_for
# 1. Import 'get_locale' from flask_babel
from flask_babel import Babel, get_locale 
from gtts import gTTS

# Initialize Flask app
app = Flask(__name__)
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'
app.secret_key = 'fasal-ai-secret-key'
babel = Babel(app)

LANGUAGES = {
    'en': 'English',
    'hi': 'हिंदी',
    'mr': 'मराठी', # Corrected script for Marathi
    'gu': 'ગુજરાતી',
    'bn': 'বাংলা'
}

# This function determines which language to use for the request
@babel.localeselector
def get_user_locale(): # Renamed to avoid confusion with the imported get_locale
    return session.get('language', request.accept_languages.best_match(LANGUAGES.keys()))

# 2. This makes the get_locale() function available in all templates
@app.context_processor
def inject_locale():
    return dict(get_locale=get_locale)

# 3. THIS IS THE FIX: This route now redirects to apply the language change.
@app.route('/change-lang/<lang>')
def change_language(lang):
    if lang in LANGUAGES:
        session['language'] = lang
    # Redirect back to the homepage to reload with the new language
    return redirect(url_for('home'))

@app.route('/speak/<text>')
def text_to_speech(text):
    # Use the selected locale for gTTS
    lang = str(get_locale())
    tts = gTTS(text=text, lang=lang)
    
    # Use an in-memory buffer to avoid creating temp files
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0) # Rewind the buffer to the beginning
    
    return send_file(mp3_fp, mimetype='audio/mpeg')

@app.route('/')
def home():
    return render_template('index.html', languages=LANGUAGES)

# --- (The rest of your code remains exactly the same) ---

# Load the trained model and label encoder
try:
    with open('crop_recommendation_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    print("✓ Model and label encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    label_encoder = None

# Crop information dictionary with growing tips
CROP_INFO = {
    'Rice': {
        'season': 'Kharif (June-November)',
        'water_req': 'High (1200-2500mm)',
        'soil_type': 'Clay loam, well-drained',
        'tips': 'Ensure proper water management and pest control for diseases like blast and brown spot.'
    },
    'Maize': {
        'season': 'Kharif/Rabi (April-July, October-January)', 
        'water_req': 'Medium (500-800mm)',
        'soil_type': 'Well-drained loamy soil',
        'tips': 'Apply balanced fertilizers and control stem borer and fall armyworm.'
    },
    'Chickpea': {
        'season': 'Rabi (October-April)',
        'water_req': 'Low (300-400mm)',
        'soil_type': 'Well-drained, neutral pH',
        'tips': 'Drought tolerant crop, avoid waterlogging and control pod borer.'
    },
    'Cotton': {
        'season': 'Kharif (May-November)',
        'water_req': 'Medium (700-1200mm)', 
        'soil_type': 'Deep, well-drained black soil',
        'tips': 'Monitor for bollworm attacks and ensure adequate potash supply.'
    },
    'Apple': {
        'season': 'Perennial (Harvest: September-November)',
        'water_req': 'Medium (1000-1200mm)',
        'soil_type': 'Well-drained, slightly acidic',
        'tips': 'Requires cold winters for dormancy. Prune regularly and control scab disease.'
    },
    'Banana': {
        'season': 'Year-round planting',
        'water_req': 'High (1500-1800mm)',
        'soil_type': 'Rich, well-drained loamy soil',
        'tips': 'High potassium requirement. Protect from strong winds and control nematodes.'
    },
    'Coffee': {
        'season': 'Perennial (Harvest: December-February)',
        'water_req': 'Medium (1500-2000mm)',
        'soil_type': 'Well-drained, slightly acidic',
        'tips': 'Shade-grown crop. Control coffee berry borer and leaf rust disease.'
    }
}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('index.html', 
                                   error="Model not loaded. Please check server configuration.",
                                   languages=LANGUAGES)

        # Get form data
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorus'])  
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Validate input ranges
        if not (0 <= N <= 200):
            return render_template('index.html', 
                                   error="Nitrogen should be between 0-200 kg/ha",
                                   languages=LANGUAGES)
        if not (0 <= P <= 150):
            return render_template('index.html',
                                   error="Phosphorus should be between 0-150 kg/ha",
                                   languages=LANGUAGES) 
        if not (0 <= K <= 300):
            return render_template('index.html',
                                   error="Potassium should be between 0-300 kg/ha",
                                   languages=LANGUAGES)
        if not (5 <= temperature <= 45):
            return render_template('index.html',
                                   error="Temperature should be between 5-45°C",
                                   languages=LANGUAGES)
        if not (10 <= humidity <= 100):
            return render_template('index.html',
                                   error="Humidity should be between 10-100%",
                                   languages=LANGUAGES)
        if not (3 <= ph <= 10):
            return render_template('index.html',
                                   error="pH should be between 3-10",
                                   languages=LANGUAGES)
        if not (20 <= rainfall <= 300):
            return render_template('index.html',
                                   error="Rainfall should be between 20-300 mm",
                                   languages=LANGUAGES)

        # Make prediction
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0]) * 100

        # Get crop information
        crop_info = CROP_INFO.get(prediction, {
            'season': 'Information not available',
            'water_req': 'Information not available', 
            'soil_type': 'Information not available',
            'tips': 'Consult local agricultural extension officer for more details.'
        })

        return render_template('index.html', 
                               prediction=prediction,
                               confidence=round(confidence, 1),
                               crop_info=crop_info,
                               languages=LANGUAGES,
                               input_values={
                                   'nitrogen': N, 'phosphorus': P, 'potassium': K,
                                   'temperature': temperature, 'humidity': humidity,
                                   'ph': ph, 'rainfall': rainfall
                               })

    except ValueError:
        return render_template('index.html', 
                               error="Please enter valid numeric values for all fields.",
                               languages=LANGUAGES)
    except Exception as e:
        return render_template('index.html',
                               error=f"An error occurred: {str(e)}",
                               languages=LANGUAGES)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.json
        features = np.array([[
            data['N'], data['P'], data['K'], 
            data['temperature'], data['humidity'], 
            data['ph'], data['rainfall']
        ]])

        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0]) * 100

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'crop_info': CROP_INFO.get(prediction, {})
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)