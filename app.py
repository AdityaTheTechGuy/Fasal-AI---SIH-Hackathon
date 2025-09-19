from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import pickle
import numpy as np
import pandas as pd
import io
from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_file
from flask_babel import Babel, get_locale
from gtts import gTTS
from data_aggregator import get_live_data, validate_coordinates

jh_district_data = pd.read_csv("jh_district_npk.csv")

def get_district_npk(district_name):
    name_norm = str(district_name).strip().lower()
    jh_district_data['district_norm'] = jh_district_data['district'].str.strip().str.lower()
    row = jh_district_data[jh_district_data['district_norm'] == name_norm]
    if not row.empty:
        return {
            "N": float(row.iloc[0]["N"]),
            "P": float(row.iloc[0]["P"]),
            "K": float(row.iloc[0]["K"]),
            "ph": float(row.iloc[0]["ph"])
        }
    return None



# Import numpy at the top level
import numpy as np

# --- Confidence display helper ---
def display_confidence_topk(probabilities, k=3, temperature=0.85):
    """
    Reweights probabilities within top-k for display.
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
    return round(primary_disp, 1), np.round(disp * 100.0, 1)# --- Case-safe crop info lookup (returns (info_dict, canonical_name)) ---
def get_crop_info_safe(name: str):
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
app.config['BABEL_DEFAULT_LOCALE'] = os.environ.get('BABEL_DEFAULT_LOCALE', 'en')
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'



# Use proper secret key handling

import logging
logging.getLogger('data_aggregator').warning("OWM key visibility check: %s", bool(os.getenv("OWM_API_KEY")))


app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    if os.environ.get('FLASK_ENV') == 'development':
        app.secret_key = 'development-key-only'
    else:
        raise ValueError("No FLASK_SECRET_KEY set for production environment")

babel = Babel(app)

LANGUAGES = {'en': 'English', 'hi': 'हिंदी', 'mr': 'मराठी', 'gu': 'ગુજરાતી', 'bn': 'বাংলা'}

@babel.localeselector
def get_user_locale():
    return session.get('language', request.accept_languages.best_match(LANGUAGES.keys()))

@app.context_processor
def inject_locale():
    return dict(get_locale=get_locale)

@app.route('/change-lang/<lang>')
def change_language(lang):
    if lang in LANGUAGES:
        session['language'] = lang
    return redirect(url_for('home'))

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


# Load model and label encoder only once when application starts
try:
    with open('crop_recommendation_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Verify it's a scikit-learn model
    if not hasattr(model, 'predict_proba'):
        print("Warning: Loaded model doesn't have predict_proba method")
        
    print(f"✅ Loaded Model Type: {type(model).__name__}")
    
    # Try to load label encoder but don't fail if it doesn't exist
    try:
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✓ Label encoder loaded successfully!")
    except FileNotFoundError:
        print("⚠️  Label encoder not found, using model's classes if available")
        label_encoder = None
        if hasattr(model, 'classes_'):
            print("✓ Using model's built-in classes")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    label_encoder = None

CROP_INFO = {
    'Rice': {'season': 'Kharif (June-November)', 'water_req': 'High (1200-2500mm)', 'soil_type': 'Clay loam, well-drained', 'tips': 'Ensure proper water management and pest control.'},
    'Maize': {'season': 'Kharif/Rabi', 'water_req': 'Medium (500-800mm)', 'soil_type': 'Well-drained loamy soil', 'tips': 'Apply balanced fertilizers and control stem borer.'},
    'Chickpea': {'season': 'Rabi (October-April)', 'water_req': 'Low (300-400mm)', 'soil_type': 'Well-drained, neutral pH', 'tips': 'Drought tolerant crop, avoid waterlogging.'},
    'Cotton': {'season': 'Kharif (May-November)', 'water_req': 'Medium (700-1200mm)', 'soil_type': 'Deep, well-drained black soil', 'tips': 'Monitor for bollworm attacks.'},
    'Apple': {'season': 'Perennial (Harvest: Sep-Nov)', 'water_req': 'Medium (1000-1200mm)', 'soil_type': 'Well-drained, slightly acidic', 'tips': 'Requires cold winters for dormancy.'},
    'Banana': {'season': 'Year-round planting', 'water_req': 'High (1500-1800mm)', 'soil_type': 'Rich, well-drained loamy soil', 'tips': 'High potassium requirement.'},
    'Coffee': {'season': 'Perennial (Harvest: Dec-Feb)', 'water_req': 'Medium (1500-2000mm)', 'soil_type': 'Well-drained, slightly acidic', 'tips': 'Shade-grown crop. Control coffee berry borer.'},
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

@app.route('/')
def home():
    return render_template('index.html', 
                           languages=LANGUAGES, 
                           prediction=None, 
                           confidence=None, 
                           crop_info=None,
                           input_values=None,
                           alternative_predictions=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('index.html', error="Model not loaded. Please check server configuration.", languages=LANGUAGES)

        input_values_form = request.form
        N = float(input_values_form['nitrogen'])
        P = float(input_values_form['phosphorus'])
        K = float(input_values_form['potassium'])
        temperature = float(input_values_form['temperature'])
        humidity = float(input_values_form['humidity'])
        ph = float(input_values_form['ph'])
        rainfall = float(input_values_form['rainfall'])

        if not (0 <= N <= 200):
            return render_template('index.html', error="Nitrogen must be between 0-200 kg/ha.", languages=LANGUAGES, input_values=input_values_form)
        if not (0 <= P <= 150):
            return render_template('index.html', error="Phosphorus must be between 0-150 kg/ha.", languages=LANGUAGES, input_values=input_values_form)
        if not (0 <= K <= 300):
            return render_template('index.html', error="Potassium must be between 0-300 kg/ha.", languages=LANGUAGES, input_values=input_values_form)
        if not (5 <= temperature <= 45):
            return render_template('index.html', error="Temperature must be between 5-45°C.", languages=LANGUAGES, input_values=input_values_form)
        if not (10 <= humidity <= 100):
            return render_template('index.html', error="Humidity must be between 10-100%.", languages=LANGUAGES, input_values=input_values_form)
        if not (3 <= ph <= 10):
            return render_template('index.html', error="pH must be between 3-10.", languages=LANGUAGES, input_values=input_values_form)
        if not (200 <= rainfall <= 3000):
            return render_template(
                'index.html',
                error="Annual rainfall must be between 200–3000 mm.",
                languages=LANGUAGES,
                input_values=input_values_form
            )


        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        probabilities = model.predict_proba(features)[0]
        all_classes = label_encoder.classes_

        # Compute display confidences (relative within top-k)
        primary_disp, all_disp = display_confidence_topk(probabilities, k=3, temperature=0.85)

        # Build predictions using display (%) for UI
        all_predictions = []
        for i, prob in enumerate(probabilities):
            crop_name = all_classes[i]
            all_predictions.append({
                'crop': crop_name,
                # use relative confidence for display
                'confidence': all_disp[i]
            })

        # Sort by display confidence descending
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        # Primary + alternatives for the template
        primary_prediction = all_predictions[0]['crop']
        primary_confidence = all_predictions[0]['confidence']  # now relative % (top-k)
        crop_info = CROP_INFO.get(primary_prediction, {})
        alternative_predictions = all_predictions[1:4]


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
            names = list(label_encoder.classes_) if label_encoder is not None else list(getattr(model, 'classes_', []))
            primary_conf, disp = display_confidence_topk(probs, k=3, temperature=0.85)
            top_idx = int(np.argmax(probs))
            primary_crop = names[top_idx]

            top3_idx = disp.argsort()[-3:][::-1]
            alts = []
            for i in top3_idx:
                if names[i] == primary_crop:
                    continue
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

        # Fetch live data (District CSV + OWM + NASA)
        live = get_live_data(lat, lon) or {}

        # --- Ensure Jharkhand district CSV values are applied when available ---
        dist_name = (live.get('district') or '').strip()
        if dist_name:
            npk_from_csv = get_district_npk(dist_name)
            if npk_from_csv:
                # overlay N/P/K/pH from your CSV
                live.update(npk_from_csv)
                # mark the source so provenance is correct later
                live['soil_source'] = 'district'
        else:
            # make the source explicit when we have nothing
            live.setdefault('soil_source', 'none')


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
        classes = list(label_encoder.classes_) if label_encoder is not None else list(getattr(model, 'classes_', []))
        top_idx = int(np.argmax(probs))
        crop = str(classes[top_idx]) if classes else str(model.predict(features)[0])

        # Farmer-friendly confidence (top-k temperature scaling)
        primary_disp, disp_vec = display_confidence_topk(probs, k=3, temperature=0.85)
        conf = round(primary_disp, 1)

        # Build two alternatives (excluding primary)
        top3_idx = np.argsort(disp_vec)[-3:][::-1]
        alts = []
        for i in top3_idx:
            if classes and classes[i] == crop:
                continue
            alts.append({'crop': classes[i] if classes else str(i),
                        'confidence': round(float(disp_vec[i]), 1)})
            if len(alts) == 2:
                break
        
        # --- FIX: Look up the crop info using the helper function ---
        crop_info, _ = get_crop_info_safe(crop)

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
            # --- FIX: Pass the populated crop_info object ---
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