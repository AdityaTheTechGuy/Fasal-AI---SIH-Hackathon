from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import pickle
import numpy as np
import pandas as pd
import io
import logging
import uuid
from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_file
from gtts import gTTS
from data_aggregator import get_live_data, validate_coordinates


# Import numpy at the top level
import numpy as np

# Crop information database
CROP_INFO = {
    'Rice': {'season': 'Kharif (June-November)', 'water_req': 'High (1200-2500mm)', 'soil_type': 'Clay loam, well-drained', 'tips': 'Ensure proper water management and pest control.'},
    'Maize': {'season': 'Kharif/Rabi (April-July, October-January)', 'water_req': 'Medium (500-800mm)', 'soil_type': 'Well-drained loamy soil', 'tips': 'Apply balanced fertilizers and control stem borer.'},
    'Chickpea': {'season': 'Rabi (October-April)', 'water_req': 'Low (300-400mm)', 'soil_type': 'Well-drained, neutral pH', 'tips': 'Drought tolerant crop, avoid waterlogging.'},
    'Cotton': {'season': 'Kharif (May-November)', 'water_req': 'Medium (700-1200mm)', 'soil_type': 'Deep, well-drained black soil', 'tips': 'Monitor for bollworm attacks and ensure adequate potash.'},
    'Apple': {'season': 'Perennial (Harvest: September-November)', 'water_req': 'Medium (1000-1200mm)', 'soil_type': 'Well-drained, slightly acidic', 'tips': 'Regular pruning and scab disease control needed.'},
    'Banana': {'season': 'Year-round planting', 'water_req': 'High (1500-1800mm)', 'soil_type': 'Rich, well-drained loamy soil', 'tips': 'High potassium requirement, protect from strong winds.'},
    'Coffee': {'season': 'Perennial (Harvest: December-February)', 'water_req': 'Medium (1500-2000mm)', 'soil_type': 'Well-drained, slightly acidic', 'tips': 'Shade-grown crop, control berry borer and leaf rust.'},
    'Kidneybeans': {'season': 'Rabi (October-March)', 'water_req': 'Medium (400-500mm)', 'soil_type': 'Well-drained loamy soil', 'tips': 'Avoid waterlogging and ensure proper nitrogen fixation.'},
    'Pigeonpeas': {'season': 'Kharif (June-December)', 'water_req': 'Medium (600-650mm)', 'soil_type': 'Well-drained sandy loam', 'tips': 'Drought tolerant, good for intercropping systems.'},
    'Mothbeans': {'season': 'Kharif (July-October)', 'water_req': 'Low (300-400mm)', 'soil_type': 'Sandy, drought-prone areas', 'tips': 'Highly drought tolerant, suitable for arid regions.'},
    'Mungbean': {'season': 'Kharif/Summer (March-June, July-October)', 'water_req': 'Medium (400-500mm)', 'soil_type': 'Well-drained sandy loam', 'tips': 'Short duration crop, good for crop rotation.'},
    'Blackgram': {'season': 'Kharif/Rabi (June-September, October-January)', 'water_req': 'Medium (400-500mm)', 'soil_type': 'Well-drained loamy soil', 'tips': 'Sensitive to waterlogging, ensure good drainage.'},
    'Lentil': {'season': 'Rabi (October-April)', 'water_req': 'Low (300-400mm)', 'soil_type': 'Well-drained sandy loam', 'tips': 'Cold tolerant, avoid excessive moisture during flowering.'},
    'Pomegranate': {'season': 'Year-round (Peak: October-February)', 'water_req': 'Medium (500-700mm)', 'soil_type': 'Well-drained, slightly alkaline', 'tips': 'Drought tolerant once established, prune regularly.'},
    'Mango': {'season': 'Perennial (Harvest: April-July)', 'water_req': 'Medium (750-1200mm)', 'soil_type': 'Well-drained, deep soil', 'tips': 'Avoid waterlogging during flowering, control fruit fly.'},
    'Grapes': {'season': 'Perennial (Harvest: February-April)', 'water_req': 'Medium (500-700mm)', 'soil_type': 'Well-drained, slightly alkaline', 'tips': 'Requires pruning and trellising, control powdery mildew.'},
    'Watermelon': {'season': 'Summer (February-May)', 'water_req': 'Medium (400-600mm)', 'soil_type': 'Sandy loam, well-drained', 'tips': 'Requires warm weather, control fruit fly and aphids.'},
    'Muskmelon': {'season': 'Summer (February-May)', 'water_req': 'Medium (300-500mm)', 'soil_type': 'Sandy loam, well-drained', 'tips': 'Warm season crop, ensure adequate potash supply.'},
    'Orange': {'season': 'Perennial (Harvest: December-February)', 'water_req': 'Medium (1000-1200mm)', 'soil_type': 'Well-drained, slightly acidic', 'tips': 'Regular irrigation needed, control citrus canker.'},
    'Papaya': {'season': 'Year-round planting', 'water_req': 'High (1200-1500mm)', 'soil_type': 'Well-drained, rich organic matter', 'tips': 'Avoid waterlogging, control papaya ring spot virus.'},
    'Coconut': {'season': 'Year-round planting', 'water_req': 'High (1500-2000mm)', 'soil_type': 'Coastal sandy soil', 'tips': 'High potash requirement, control rhinoceros beetle.'},
    'Jute': {'season': 'Kharif (April-August)', 'water_req': 'High (1000-1500mm)', 'soil_type': 'Alluvial soil with high moisture', 'tips': 'Requires high humidity, harvest at proper maturity.'}
}

# --- Updated Confidence display helper for better sensitivity ---
def display_confidence_topk(probabilities, k=3, temperature=1.0):
    """
    Updated confidence display function with better sensitivity.
    Reduced default temperature from 0.85 to 1.0 for more natural probability display.
    T < 1 sharpens peaks, T > 1 softens.
    Returns (primary_conf, all_conf_list)
    """
    # Convert input to numpy array
    probs = np.array(probabilities, dtype=float)
    probs = np.maximum(probs, 1e-12)

    if temperature and float(temperature) > 0:
        T = float(temperature)
        probs = probs ** (1.0 / T)
        probs = probs / probs.sum()

    # Get indices sorted by probability (descending)
    sorted_idx = np.argsort(probs)[::-1]  # Reverse to get descending order

    # Take top k indices
    topk_idx = sorted_idx[:k]
    topk_sum = probs[topk_idx].sum()

    disp = np.zeros_like(probs)
    if topk_sum > 0:
        disp[topk_idx] = probs[topk_idx] / topk_sum

    top_idx = int(sorted_idx[0])  # Most probable class
    primary_disp = float(disp[top_idx] * 100.0)

    # Return a NumPy array instead of a list
    return round(primary_disp, 1), np.round(disp * 100.0, 1)

# --- Case-safe crop info lookup (returns (info_dict, canonical_name)) ---
def findCropEntry(name: str):
    """
    Finds a crop entry in the CROP_INFO database with case-insensitive matching.
    """
    n = str(name).strip().lower()
    for key, info in CROP_INFO.items():
        if key.lower() == n:
            return info, key  # return canonical dictionary key for display

    # Fallback copy so we never mutate the global template
    return {
        'season': 'Season information not available',
        'water_req': 'Water requirement not available',
        'soil_type': 'Soil type not available',
        'tips': 'Growing tips not available'
    }, name

app = Flask(__name__)
# Use proper secret key handling
logging.getLogger('data_aggregator').warning("OWM key visibility check: %s", bool(os.getenv("OWM_API_KEY")))

app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    if os.environ.get('FLASK_ENV') == 'development':
        app.secret_key = 'development-key-only'
    else:
        raise ValueError("No FLASK_SECRET_KEY set for production environment")


LANGUAGES = {
    'en': 'English',
    'hi': 'हिंदी',
    'gu': 'ગુજરાતી',
    'bn': 'বাংলা',
    'or': 'ଓଡିଆ',
    'bho': 'भोजपुरी'
}

def get_locale():
    return 'en'  # Default to English, Google Translate handles translations

@app.context_processor
def inject_helpers():
    # Inject helpers into the template context
    if '_csrf_token' not in session:
        session['_csrf_token'] = str(uuid.uuid4())
    return dict(
        get_locale=get_locale,
        findCropEntry=findCropEntry,
        csrf_token=session['_csrf_token']
    )

@app.route('/change-lang/<lang>')
def change_language(lang):
    if lang in LANGUAGES:
        session['lang'] = lang
        session.modified = True
    else:
        session['lang'] = 'en'
        session.modified = True
    return redirect(request.referrer or url_for('home'))

@app.route('/speak/<text>')
def text_to_speech(text):
    lang = str(get_locale())
    tts = gTTS(text=text, lang=lang)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return send_file(mp3_fp, mimetype='audio/mpeg')

# Expected feature order for model prediction
EXPECTED_FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Load model with enhanced error handling and metadata support
model = None
model_data = None
try:
    model_path = os.path.join(os.path.dirname(__file__), 'crop_recommendation_model.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Check if it's the new enhanced model format with metadata
    if isinstance(model_data, dict) and 'model' in model_data:
        print("✅ Loading enhanced model with metadata...")
        model = model_data['model']
        feature_names = model_data.get('feature_names', EXPECTED_FEATURES)
        feature_importance = model_data.get('feature_importance', {})
        crop_labels = model_data.get('crop_labels', [])

        print(f"✅ Model type: {type(model).__name__}")
        print(f"✅ Features: {feature_names}")
        print(f"✅ Available crops: {len(crop_labels)}")

    else:
        # Legacy model format
        print("⚠️  Loading legacy model format...")
        model = model_data
        feature_names = EXPECTED_FEATURES
        feature_importance = {}
        crop_labels = []

    # Verify it's a scikit-learn model
    if not hasattr(model, 'predict_proba'):
        print("Warning: Loaded model doesn't have predict_proba method")

    print(f"✅ Loaded Model Type: {type(model).__name__}")

except FileNotFoundError:
    print("❌ Model file not found. Please train the model first using train_model_updated.py")
    model = None
    feature_names = EXPECTED_FEATURES
    feature_importance = {}
    crop_labels = []
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    feature_names = EXPECTED_FEATURES
    feature_importance = {}
    crop_labels = []

# Try to load label encoder but don't fail if it doesn't exist
try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
        print("✓ Label encoder loaded successfully!")
except FileNotFoundError:
    print("⚠️ Label encoder not found, using model's classes if available")
    label_encoder = None
    if hasattr(model, 'classes_'):
        print("✓ Using model's built-in classes")

@app.route('/')
def home():
    # Ensure CSRF token exists for the session
    if '_csrf_token' not in session:
        session['_csrf_token'] = str(uuid.uuid4())
    return render_template('index.html',
                         languages=LANGUAGES,
                         prediction=None,
                         confidence=None,
                         crop_info=None,
                         input_values=None,
                         alternative_predictions=None,
                         confidence_label='Confidence Level',
                         error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # CSRF Token Validation
        submitted_token = request.form.get('csrf_token')
        if not submitted_token or submitted_token != session.get('_csrf_token'):
            return render_template('index.html', error="Invalid form submission. Please try again.", languages=LANGUAGES)

        if model is None:
            return render_template('index.html',
                                 error="Model not loaded. Please train the model using train_model_updated.py",
                                 languages=LANGUAGES)

        input_values_form = request.form

        N = float(input_values_form['nitrogen'])
        P = float(input_values_form['phosphorus'])
        K = float(input_values_form['potassium'])
        temperature = float(input_values_form['temperature'])
        humidity = float(input_values_form['humidity'])
        ph = float(input_values_form['ph'])
        rainfall = float(input_values_form['rainfall'])

        # Input validation
        if not (0 <= N <= 200):
            return render_template('index.html', error="Nitrogen must be between 0-200 kg/ha.",
                                 languages=LANGUAGES, input_values=input_values_form)
        if not (0 <= P <= 150):
            return render_template('index.html', error="Phosphorus must be between 0-150 kg/ha.",
                                 languages=LANGUAGES, input_values=input_values_form)
        if not (0 <= K <= 300):
            return render_template('index.html', error="Potassium must be between 0-300 kg/ha.",
                                 languages=LANGUAGES, input_values=input_values_form)
        if not (5 <= temperature <= 45):
            return render_template('index.html', error="Temperature must be between 5-45°C.",
                                 languages=LANGUAGES, input_values=input_values_form)
        if not (10 <= humidity <= 100):
            return render_template('index.html', error="Humidity must be between 10-100%.",
                                 languages=LANGUAGES, input_values=input_values_form)
        if not (3 <= ph <= 10):
            return render_template('index.html', error="pH must be between 3-10.",
                                 languages=LANGUAGES, input_values=input_values_form)
        if not (200 <= rainfall <= 3000):
            return render_template('index.html',
                                 error="Annual rainfall must be between 200–3000 mm.",
                                 languages=LANGUAGES,
                                 input_values=input_values_form)

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        probabilities = model.predict_proba(features)[0]

        # Get classes - handle both legacy and enhanced model formats
        if hasattr(model, 'classes_'):
            all_classes = model.classes_
        elif crop_labels:
            all_classes = crop_labels
        elif label_encoder and hasattr(label_encoder, 'classes_'):
            all_classes = label_encoder.classes_
        else:
            # Fallback for very old models
            all_classes = list(CROP_INFO.keys())

        # Compute display confidences with improved sensitivity (temperature=1.0 instead of 0.85)
        primary_disp, all_disp = display_confidence_topk(probabilities, k=3, temperature=1.0)

        # Build predictions using display (%) for UI
        all_predictions = []
        for i, prob in enumerate(probabilities):
            if i < len(all_classes):
                crop_name = all_classes[i]
                all_predictions.append({
                    'crop': crop_name,
                    'confidence': all_disp[i]  # use relative confidence for display
                })

        # Sort by display confidence descending
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        # Primary + alternatives for the template
        primary_prediction = all_predictions[0]['crop']
        primary_confidence = all_predictions[0]['confidence']  # now relative % (top-k)

        crop_info, _ = findCropEntry(primary_prediction)
        alternative_predictions = all_predictions[1:4]

        # Regenerate CSRF token after successful processing
        session['_csrf_token'] = str(uuid.uuid4())

        return render_template('index.html',
                             prediction=primary_prediction,
                             confidence=primary_confidence,
                             crop_info=crop_info,
                             alternative_predictions=alternative_predictions,
                             languages=LANGUAGES,
                             input_values=input_values_form)

    except ValueError:
        return render_template('index.html', error="Please enter valid numeric values.", languages=LANGUAGES)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}", languages=LANGUAGES)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True) or {}
    features = {}

    # Require all parameters to be present
    for key in ['N','P','K','temperature','humidity','ph','rainfall']:
        val = data.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            return jsonify({'status': 'error', 'message': f'Missing required parameter: {key}'}), 400

        try:
            features[key] = float(val)
        except ValueError:
            return jsonify({'status': 'error', 'message': f'Invalid value for {key}: must be a number'}), 400

    # Validate input ranges
    if not (0 <= features['N'] <= 200):
        return jsonify({'status': 'error', 'message': 'Nitrogen must be between 0-200 kg/ha.'}), 400
    if not (0 <= features['P'] <= 150):
        return jsonify({'status': 'error', 'message': 'Phosphorus must be between 0-150 kg/ha.'}), 400
    if not (0 <= features['K'] <= 300):
        return jsonify({'status': 'error', 'message': 'Potassium must be between 0-300 kg/ha.'}), 400
    if not (5 <= features['temperature'] <= 45):
        return jsonify({'status': 'error', 'message': 'Temperature must be between 5-45°C.'}), 400
    if not (10 <= features['humidity'] <= 100):
        return jsonify({'status': 'error', 'message': 'Humidity must be between 10-100%.'}), 400
    if not (3 <= features['ph'] <= 10):
        return jsonify({'status': 'error', 'message': 'pH must be between 3-10.'}), 400
    if not (200 <= features['rainfall'] <= 3000):
        return jsonify({'status': 'error', 'message': 'Annual rainfall must be between 200-3000mm.'}), 400

    # Clamp values to safe ranges
    features['humidity'] = min(100.0, max(10.0, features['humidity']))
    features['ph'] = min(10.0, max(3.0, features['ph']))

    if model is None:
        return jsonify({'status':'error','message':'Model not loaded.'}), 500

    X = np.array([[features['N'], features['P'], features['K'],
                   features['temperature'], features['humidity'],
                   features['ph'], features['rainfall']]], dtype=float)

    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[0]

            # Get class names with proper fallback
            if hasattr(model, 'classes_'):
                names = list(model.classes_)
            elif crop_labels:
                names = crop_labels
            elif label_encoder and hasattr(label_encoder, 'classes_'):
                names = list(label_encoder.classes_)
            else:
                names = list(CROP_INFO.keys())

            # Use improved confidence display (temperature=1.0)
            primary_conf, disp = display_confidence_topk(probs, k=3, temperature=1.0)

            top_idx = int(np.argmax(probs))
            primary_crop = names[top_idx] if top_idx < len(names) else 'Unknown'

            top3_idx = disp.argsort()[-3:][::-1]
            alts = []
            for i in top3_idx:
                if i < len(names) and names[i] != primary_crop:
                    alts.append({'crop': names[i], 'confidence': round(float(disp[i]), 1)})
                if len(alts) == 2:
                    break

            return jsonify({
                'status':'success',
                'crop': primary_crop,
                'confidence': round(float(primary_conf), 1),
                'alternatives': alts,
                'environment_data': features,
                'location_data': {},
                'crop_info': {}
            })
        else:
            pred = model.predict(X)[0]
            return jsonify({'status':'success','crop': pred,'confidence': 100.0,
                          'alternatives': [],'environment_data': features,
                          'location_data': {},'crop_info': {}})

    except Exception as e:
        return jsonify({'status':'error','message': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json(silent=True) or {}
        
        # CSRF Token Validation for AJAX
        submitted_token = data.get('csrf_token')
        if not submitted_token or submitted_token != session.get('_csrf_token'):
            return jsonify({'status': 'error', 'message': 'CSRF validation failed.'}), 400

        user_message = (data.get('message') or '').strip()
        history = data.get('history') or []  # optional: list of {role, content}

        if not user_message:
            return jsonify({'status': 'error', 'message': 'Message is required.'}), 400

        # Configure Gemini client
        try:
            import google.generativeai as genai
        except Exception as e:
            return jsonify({'status': 'error', 'message': 'Gemini client not installed. Please install google-generativeai.', 'detail': str(e)}), 500

        gemini_api_key = os.environ.get('GEMINI_API_KEY') or 'AIzaSyDUcStsKHm-BZ4hvf6O9vdDdKjHaxgv1iQ'
        if not gemini_api_key:
            return jsonify({'status': 'error', 'message': 'GEMINI_API_KEY is not configured.'}), 500

        genai.configure(api_key=gemini_api_key)

        # Prepare a system prompt to keep answers on agriculture domain
        system_prompt = (
            "You are FasalAI, an assistant for farmers. Provide concise, helpful answers "
            "about crops, soil, weather, irrigation, pests, and best practices for Indian agriculture. "
            "If asked unrelated questions, briefly respond and guide back to farming context."
        )

        model = genai.GenerativeModel('gemini-1.5-flash')

        # Build messages: optional history, then system + user
        contents = []
        if isinstance(history, list):
            for turn in history[-10:]:
                role = (turn.get('role') or '').lower()
                content = (turn.get('content') or '').strip()
                if not content:
                    continue
                if role in ('user', 'model', 'assistant'):
                    mapped_role = 'user' if role == 'user' else 'model'
                    contents.append({ 'role': mapped_role, 'parts': [content] })

        # Inject system guidance at the front
        contents.insert(0, { 'role': 'user', 'parts': [system_prompt] })
        contents.append({ 'role': 'user', 'parts': [user_message] })

        response = model.generate_content(contents)
        text = (getattr(response, 'text', None) or '').strip()

        if not text and hasattr(response, 'candidates'):
            # Fallback: extract first candidate text
            try:
                text = (response.candidates[0].content.parts[0].text or '').strip()
            except Exception:
                text = ''

        if not text:
            return jsonify({'status': 'error', 'message': 'No response from model.'}), 502

        return jsonify({
            'status': 'success',
            'reply': text
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict_live', methods=['POST'])
def predict_live():
    try:
        if model is None:
            return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500

        data = request.get_json(silent=True) or {}
        if 'latitude' not in data or 'longitude' not in data:
            return jsonify({'status': 'error', 'message': 'Latitude and longitude are required.'}), 400

        # Coordinates
        try:
            lat = float(data['latitude'])
            lon = float(data['longitude'])
        except Exception:
            return jsonify({'status': 'error', 'message': 'Latitude/longitude must be numeric.'}), 400

        if not validate_coordinates(lat, lon):
            return jsonify({'status': 'error', 'message': 'Invalid coordinates.'}), 400

        # Fetch live data (OWM + NASA)
        live = get_live_data(lat, lon) or {}

        # Validate and convert all required values
        vals = {}
        required_fields = ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']

        for key in required_fields:
            v = live.get(key)
            if v is None:
                return jsonify({'status': 'error',
                              'message': f'Missing required live data: {key}'}), 400
            try:
                vals[key] = float(v)
            except ValueError:
                return jsonify({'status': 'error',
                              'message': f'Invalid value for {key}: must be a number'}), 400

        # Clamp
        vals['humidity'] = min(100.0, max(0.0, vals['humidity']))
        vals['ph'] = min(10.0, max(3.0, vals['ph']))

        # Optional legacy rescale
        if os.environ.get('RAINFALL_RESCALE','0') == '1':
            vals['rainfall'] = vals['rainfall'] / 12.0

        # Feature vector for model (strict order)
        features = np.array([[vals['N'], vals['P'], vals['K'],
                             vals['temperature'], vals['humidity'],
                             vals['ph'], vals['rainfall']]], dtype=float)

        # Predict class + probabilities
        probs = model.predict_proba(features)[0]

        # Get class names with proper fallback
        if hasattr(model, 'classes_'):
            classes = list(model.classes_)
        elif crop_labels:
            classes = crop_labels
        elif label_encoder and hasattr(label_encoder, 'classes_'):
            classes = list(label_encoder.classes_)
        else:
            classes = list(CROP_INFO.keys())

        top_idx = int(np.argmax(probs))
        crop = str(classes[top_idx]) if top_idx < len(classes) else str(model.predict(features)[0])

        # Farmer-friendly confidence with improved sensitivity (temperature=1.0)
        primary_disp, disp_vec = display_confidence_topk(probs, k=3, temperature=1.0)
        conf = round(primary_disp, 1)

        # Build two alternatives (excluding primary)
        top3_idx = np.argsort(disp_vec)[-3:][::-1]
        alts = []
        for i in top3_idx:
            if i < len(classes) and classes[i] != crop:
                alts.append({'crop': classes[i],
                            'confidence': round(float(disp_vec[i]), 1)})
            if len(alts) == 2:
                break

        # Look up the crop info using the helper function
        crop_info, _ = findCropEntry(crop)

        return jsonify({
            'status': 'success',
            'crop': crop,
            'confidence': conf,
            'alternatives': alts,
            'environment_data': {
                'N': vals['N'], 'P': vals['P'], 'K': vals['K'],
                'temperature': vals['temperature'], 'humidity': vals['humidity'],
                'ph': vals['ph'], 'rainfall': vals['rainfall']
            },
            'location_data': {
                'latitude': lat,
                'longitude': lon,
                'district': live.get('district')
            },
            'crop_info': crop_info
        })

    except Exception as e:
        try:
            dbg = {'incoming_live': live if 'live' in locals() else None}
        except Exception:
            dbg = None
        return jsonify({'status': 'error', 'message': str(e), 'debug': dbg}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
