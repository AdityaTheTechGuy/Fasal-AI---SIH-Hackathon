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

# --- Confidence display helper ---
def display_confidence_topk(probabilities, k=3, temperature=0.85):
    """
    Reweights probabilities within top-k for display.
    T < 1 sharpens peaks, T > 1 softens.
    Returns (primary_conf, all_conf_list)
    """
    import numpy as np
    probs = np.asarray(probabilities, dtype=float)
    probs = np.maximum(probs, 1e-12)

    if temperature and float(temperature) > 0:
        T = float(temperature)
        probs = probs ** (1.0 / T)
        probs = probs / probs.sum()

    # Focus only on top-k
    topk_idx = np.argsort(probs)[-k:]
    topk_sum = probs[topk_idx].sum()

    disp = np.zeros_like(probs)
    if topk_sum > 0:
        disp[topk_idx] = probs[topk_idx] / topk_sum

    top_idx = int(np.argmax(probs))
    primary_disp = float(disp[top_idx] * 100.0)
    return round(primary_disp, 1), [round(x * 100.0, 1) for x in disp]

# --- Provenance helper: returns (float_value, "live"|"default") ---
def coerce_float_with_provenance(val, default):
    """
    Try to cast val->float. If invalid or NaN, return default and mark source="default".
    """
    try:
        v = float(val)
        if v != v:  # NaN
            raise ValueError("NaN")
        return v, "live"
    except Exception:
        return float(default), "default"
    
    # --- Case-safe crop info lookup (returns (info_dict, canonical_name)) ---
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
        raise ValueError("Loaded model doesn't have predict_proba method - may not be a scikit-learn model")
        
    print(f"✅ Loaded Model Type: {type(model).__name__}")
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("✓ Model and label encoder loaded successfully!")
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
        if not (20 <= rainfall <= 300):
            return render_template('index.html', error="Rainfall must be between 20-300 mm.", languages=LANGUAGES, input_values=input_values_form)

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
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.json
        
        # Validate input ranges
        if not (0 <= data['N'] <= 200):
            return jsonify({'error': 'Nitrogen must be between 0-200 kg/ha'}), 400
        if not (0 <= data['P'] <= 150):
            return jsonify({'error': 'Phosphorus must be between 0-150 kg/ha'}), 400
        if not (0 <= data['K'] <= 300):
            return jsonify({'error': 'Potassium must be between 0-300 kg/ha'}), 400
        if not (5 <= data['temperature'] <= 45):
            return jsonify({'error': 'Temperature must be between 5-45°C'}), 400
        if not (10 <= data['humidity'] <= 100):
            return jsonify({'error': 'Humidity must be between 10-100%'}), 400
        if not (3 <= data['ph'] <= 10):
            return jsonify({'error': 'pH must be between 3-10'}), 400
        if not (20 <= data['rainfall'] <= 300):
            return jsonify({'error': 'Rainfall must be between 20-300 mm'}), 400

        features = np.array([[
            data['N'], data['P'], data['K'], 
            data['temperature'], data['humidity'], 
            data['ph'], data['rainfall']
        ]])
        
        # Get all predictions with probabilities
        probabilities = model.predict_proba(features)[0]
        all_classes = label_encoder.classes_

        # Compute relative confidences
        primary_disp, all_disp = display_confidence_topk(probabilities, k=3, temperature=0.85)

        all_predictions = []
        for i, prob in enumerate(probabilities):
            crop_name = all_classes[i]
            all_predictions.append({
                'crop': crop_name,
                'confidence': round(float(prob * 100), 1),       # raw %
                'display_confidence': all_disp[i],               # relative %
                'crop_info': CROP_INFO.get(crop_name, {})
            })

        # Sort by display_confidence (farmer-friendly ranking)
        all_predictions.sort(key=lambda x: x['display_confidence'], reverse=True)

        return jsonify({
            'primary_prediction': all_predictions[0],
            'alternative_predictions': all_predictions[1:4],
            'all_predictions': all_predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# NEW: Live prediction endpoint using geolocation and real-time data (final)
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

        # Fetch live data from external APIs (OWM / SoilGrids inside data_aggregator)
        live = get_live_data(lat, lon)
        if not live:
            return jsonify({'status': 'error', 'message': 'Could not retrieve live data.'}), 503

        # ---- sanitize EVERYTHING to pure floats + provenance ----
        vals = {}
        provenance = {}

        vals['N'],           provenance['N']           = coerce_float_with_provenance(live.get('N'),           50.0)
        vals['P'],           provenance['P']           = coerce_float_with_provenance(live.get('P'),           40.0)
        vals['K'],           provenance['K']           = coerce_float_with_provenance(live.get('K'),           200.0)
        vals['temperature'], provenance['temperature'] = coerce_float_with_provenance(live.get('temperature'), 25.0)
        vals['humidity'],    provenance['humidity']    = coerce_float_with_provenance(live.get('humidity'),    60.0)
        vals['ph'],          provenance['ph']          = coerce_float_with_provenance(live.get('ph'),          6.5)
        vals['rainfall'],    provenance['rainfall']    = coerce_float_with_provenance(live.get('rainfall'),    0.0)

        # Feature vector for model
        features = np.array([[vals['N'], vals['P'], vals['K'],
                              vals['temperature'], vals['humidity'],
                              vals['ph'], vals['rainfall']]], dtype=float)

        # Predict class + probabilities
        probs = model.predict_proba(features)[0]
        classes = getattr(label_encoder, 'classes_', getattr(model, 'classes_', None))
        top_idx = int(np.argmax(probs))
        crop = str(classes[top_idx]) if classes is not None else str(model.predict(features)[0])

        # Use relative (top-k) confidence for farmer-facing UI
        primary_disp, _ = display_confidence_topk(probs, k=3, temperature=0.85)
        conf = round(primary_disp, 1)

        # Overall data quality summary
        defaults_used = [k for k, src in provenance.items() if src == "default"]
        data_quality = (
            "live" if not defaults_used else
            "mixed" if len(defaults_used) < len(provenance) else
            "default-only"
        )

        # Attach crop info (case-safe) and use canonical name for display
        crop_info, canonical_name = get_crop_info_safe(crop)

        return jsonify({
            'status': 'success',
            'crop': canonical_name,   # show "Muskmelon" instead of "muskmelon"
            'confidence': conf,
            'crop_info': crop_info,
            'environment_data': vals,
            'provenance': provenance,
            'data_quality': data_quality,
            'defaults_used': defaults_used,
            'location_data': {'latitude': lat, 'longitude': lon}
        })


    except Exception as e:
        # Optional debug to help diagnose data shape/type issues quickly
        dbg = None
        try:
            dbg = {'incoming_live': live if 'live' in locals() else None,
                   'provenance': provenance if 'provenance' in locals() else None}
        except Exception:
            pass
        return jsonify({'status': 'error', 'message': str(e), 'debug': dbg}), 500




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)