// FasalAI - AI Crop Recommendation System JavaScript

// Crop database with detailed information
const cropDatabase = {
    "Rice": { name: "Rice", season: "Kharif (June-November)", water_req: "High (1200-2500mm)", soil_type: "Clay loam, well-drained", tips: "Ensure proper water management and pest control for diseases like blast and brown spot.", icon: "🌾" },
    "Maize": { name: "Maize", season: "Kharif/Rabi (April-July, October-January)", water_req: "Medium (500-800mm)", soil_type: "Well-drained loamy soil", tips: "Apply balanced fertilizers and control stem borer and fall armyworm.", icon: "🌽" },
    "Chickpea": { name: "Chickpea", season: "Rabi (October-April)", water_req: "Low (300-400mm)", soil_type: "Well-drained, neutral pH", tips: "Drought tolerant crop, avoid waterlogging and control pod borer.", icon: "🧆" },
    "Cotton": { name: "Cotton", season: "Kharif (May-November)", water_req: "Medium (700-1200mm)", soil_type: "Deep, well-drained black soil", tips: "Monitor for bollworm attacks and ensure adequate potash supply.", icon: "🧶" },
    "Apple": { name: "Apple", season: "Perennial (Harvest: September-November)", water_req: "Medium (1000-1200mm)", soil_type: "Well-drained, slightly acidic", tips: "Requires cold winters for dormancy. Prune regularly and control scab disease.", icon: "🍎" },
    "Banana": { name: "Banana", season: "Year-round planting", water_req: "High (1500-1800mm)", soil_type: "Rich, well-drained loamy soil", tips: "High potassium requirement. Protect from strong winds and control nematodes.", icon: "🍌" },
    "Coffee": { name: "Coffee", season: "Perennial (Harvest: December-February)", water_req: "Medium (1500-2000mm)", soil_type: "Well-drained, slightly acidic", tips: "Shade-grown crop. Control coffee berry borer and leaf rust disease.", icon: "☕️" },
    "Kidneybeans": { name: "Kidneybeans", season: "Rabi (October-March)", water_req: "Medium (400-500mm)", soil_type: "Well-drained loamy soil", tips: "Avoid waterlogging and ensure proper nitrogen fixation.", icon: "🌰" },
    "Pigeonpeas": { name: "Pigeonpeas", season: "Kharif (June-December)", water_req: "Medium (600-650mm)", soil_type: "Well-drained sandy loam", tips: "Drought tolerant, good for intercropping systems.", icon: "🫛" },
    "Mothbeans": { name: "Mothbeans", season: "Kharif (July-October)", water_req: "Low (300-400mm)", soil_type: "Sandy, drought-prone areas", tips: "Highly drought tolerant, suitable for arid regions.", icon: "🟫" },
    "Mungbean": { name: "Mungbean", season: "Kharif/Summer (March-June, July-October)", water_req: "Medium (400-500mm)", soil_type: "Well-drained sandy loam", tips: "Short duration crop, good for crop rotation.", icon: "🟢" },
    "Blackgram": { name: "Blackgram", season: "Kharif/Rabi (June-September, October-January)", water_req: "Medium (400-500mm)", soil_type: "Well-drained loamy soil", tips: "Sensitive to waterlogging, ensure good drainage.", icon: "⚫" },
    "Lentil": { name: "Lentil", season: "Rabi (October-April)", water_req: "Low (300-400mm)", soil_type: "Well-drained sandy loam", tips: "Cold tolerant, avoid excessive moisture during flowering.", icon: "🟠" },
    "Pomegranate": { name: "Pomegranate", season: "Year-round (Peak: October-February)", water_req: "Medium (500-700mm)", soil_type: "Well-drained, slightly alkaline", tips: "Drought tolerant once established, prune regularly.", icon: "🍒" },
    "Mango": { name: "Mango", season: "Perennial (Harvest: April-July)", water_req: "Medium (750-1200mm)", soil_type: "Well-drained, deep soil", tips: "Avoid waterlogging during flowering, control fruit fly.", icon: "🥭" },
    "Grapes": { name: "Grapes", season: "Perennial (Harvest: February-April)", water_req: "Medium (500-700mm)", soil_type: "Well-drained, slightly alkaline", tips: "Requires pruning and trellising, control powdery mildew.", icon: "🍇" },
    "Watermelon": { name: "Watermelon", season: "Summer (February-May)", water_req: "Medium (400-600mm)", soil_type: "Sandy loam, well-drained", tips: "Requires warm weather, control fruit fly and aphids.", icon: "🍉" },
    "Muskmelon": { name: "Muskmelon", season: "Summer (February-May)", water_req: "Medium (300-500mm)", soil_type: "Sandy loam, well-drained", tips: "Warm season crop, ensure adequate potash supply.", icon: "🍈" },
    "Orange": { name: "Orange", season: "Perennial (Harvest: December-February)", water_req: 'Medium (1000-1200mm)', soil_type: "Well-drained, slightly acidic", tips: "Regular irrigation needed, control citrus canker.", icon: "🍊" },
    "Papaya": { name: "Papaya", season: "Year-round planting", water_req: "High (1200-1500mm)", soil_type: "Well-drained, rich organic matter", tips: "Avoid waterlogging, control papaya ring spot virus.", icon: "🥝" },
    "Coconut": { name: "Coconut", season: "Year-round planting", water_req: "High (1500-2000mm)", soil_type: "Coastal sandy soil", tips: "High potash requirement, control rhinoceros beetle.", icon: "🥥" },
    "Jute": { name: "Jute", season: "Kharif (April-August)", water_req: "High (1000-1500mm)", soil_type: "Alluvial soil with high moisture", tips: "Requires high humidity, harvest at proper maturity.", icon: "🧵" }
};

// Case-insensitive lookup for crop entries
function findCropEntry(name) {
    if (!name) return null;
    const exact = cropDatabase[name];
    if (exact) return exact;
    const lower = String(name).toLowerCase();
    for (const key of Object.keys(cropDatabase)) {
        if (key.toLowerCase() === lower) return cropDatabase[key];
    }
    // Try singular/plural simple fix (remove/add trailing 's')
    if (lower.endsWith('s')) {
        const singular = lower.slice(0, -1);
        for (const key of Object.keys(cropDatabase)) {
            if (key.toLowerCase() === singular) return cropDatabase[key];
        }
    } else {
        const plural = lower + 's';
        for (const key of Object.keys(cropDatabase)) {
            if (key.toLowerCase() === plural) return cropDatabase[key];
        }
    }
    return null;
}

const keyLabels = {
  N: "Nitrogen (N)",
  P: "Phosphorus (P)",
  K: "Potassium (K)",
  ph: "Soil pH",
  temperature: "Air temperature",
  humidity: "Humidity",
  rainfall: "Annual Rainfall (mm)"
};


let suitabilityChart = null;
let predictionTimeout = null;

// --- Helpers for live recommendation UI -----------------------------

function formatLatLng(lat, lon) {
  const latAbs = Math.abs(Number(lat)).toFixed(4);
  const lonAbs = Math.abs(Number(lon)).toFixed(4);
  const latHem = Number(lat) >= 0 ? "N" : "S";
  const lonHem = Number(lon) >= 0 ? "E" : "W";
  return `${latAbs}°${latHem}, ${lonAbs}°${lonHem}`;
}

function badgeForDataQuality() {
  return `<span class="status status--success" style="margin-left:8px">Live data</span>`;
}

/**
 * Renders the entire Live result card.
 * Expects payload from /predict_live with:
 *  crop, confidence, crop_info{season,water_req,soil_type,tips},
 *  data_quality, defaults_used[], location_data{latitude,longitude}
 */
function renderLiveResult(container, data) {
  const liveResult = container;
  const provNote = `<div class="location-info" style="margin-top:8px">
       <p><i class="fas fa-circle-check"></i> Using live soil &amp; weather data</p>
     </div>`;


  const crop = data.crop || "—";
  const ci = data.crop_info || {};
  const lat = data.location_data?.latitude;
  const lon = data.location_data?.longitude;

  const db = findCropEntry(crop) || {};
  const icon = db.icon || "🌱";

  liveResult.classList.add("active");
  liveResult.innerHTML = `
    <div class="live-prediction-success">
      <div class="success-header">
        <h3>Live Recommendation ${badgeForDataQuality()}</h3>
        <div class="confidence-badge">${(data.confidence ?? 0)}% Confidence</div>
      </div>

      <div class="crop-result">
        <div class="crop-icon-large">${icon}</div>

        <div class="crop-details">
          <h2>${crop}</h2>

          <div class="crop-info">
            <p><strong>Season:</strong> ${ci.season || "—"}</p>
            <p><strong>Water Requirement:</strong> ${ci.water_req || "—"}</p>
            <p><strong>Soil Type:</strong> ${ci.soil_type || "—"}</p>
          </div>

          <div class="crop-tips">
            <h4><i class="fas fa-lightbulb"></i> Growing Tips</h4>
            <p>${ci.tips || "—"}</p>
          </div>

          ${provNote}
        </div>
      </div>

      <div class="location-info" style="margin-top:12px">
        <p><i class="fas fa-location-dot"></i>
          Location: ${formatLatLng(lat, lon)}
        </p>
        <p><i class="fas fa-database"></i>
          Data Sources: OpenWeatherMap + SoilGrids API
        </p>
      </div>
    </div>
  `;
}



// Initialize the application
document.addEventListener('DOMContentLoaded', function () {
    initializeSliders();
    setupEventListeners();
    initializeRealTimePredictions();
    setupLivePrediction();
    setupChatbot();
});

// Initialize all sliders and input synchronization
function initializeSliders() {
    const parameters = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall'];

    parameters.forEach(param => {
        const slider = document.getElementById(param);
        const input = document.getElementById(param + '-input');

        if (slider && input) {
            // Sync slider to input
            slider.addEventListener('input', function () {
                input.value = this.value;
                scheduleRealTimePrediction();
            });

            // Sync input to slider
            input.addEventListener('input', function () {
                const value = parseFloat(this.value);
                const min = parseFloat(slider.min);
                const max = parseFloat(slider.max);

                if (value >= min && value <= max) {
                    slider.value = value;
                    scheduleRealTimePrediction();
                }
            });
        }
    });
}

// Setup additional event listeners
function setupEventListeners() {
    // Add AJAX form submission
    const form = document.getElementById('crop-prediction-form');
    if (form) {
        form.addEventListener('submit', function (e) {
            e.preventDefault();
            submitForm();
        });
    }
}

// Setup live prediction functionality
function setupLivePrediction() {
    const livePredictionBtn = document.getElementById('getLocationBtn');
    if (livePredictionBtn) {
        livePredictionBtn.addEventListener('click', handleLivePrediction);
    }
}

// Handle live prediction using geolocation
async function handleLivePrediction() {
    const resultDiv = document.getElementById('live-result');
    if (!resultDiv) return;

    // Show loading state
    resultDiv.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i> Fetching your location...</div>';
    resultDiv.className = 'live-result active';

    try {
        // Get user's location
        const position = await getCurrentPosition();
        const { latitude, longitude } = position.coords;

        // Update UI
        resultDiv.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i> Location found! Analyzing soil and climate data...</div>';

        // Make prediction request
        const response = await fetch('/predict_live', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                latitude: parseFloat(latitude).toFixed(6), 
                longitude: parseFloat(longitude).toFixed(6) 
            })
        });

        let data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || 'Failed to get prediction for your location');
        }

        if (data.status === 'success') {
            // Use the unified renderer (adds data-quality badge + defaults list + crop info)
            renderLiveResult(resultDiv, data);
        } else {
            // Display error from backend
            resultDiv.innerHTML = `
                <div class="live-prediction-error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>Error</h3>
                    <p>${data.message || 'An error occurred while processing your request.'}</p>
                </div>
            `;
        }

    } catch (error) {
        console.error('Live prediction error:', error);
        resultDiv.innerHTML = `
            <div class="live-prediction-error">
                <i class="fas fa-exclamation-triangle"></i>
                <h3>Error</h3>
                <p>${error.message || 'An error occurred while processing your request.'}</p>
            </div>
        `;
    }
}

// Get current position with Promise
function getCurrentPosition() {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
            reject(new Error('Geolocation is not supported by your browser.'));
            return;
        }

        navigator.geolocation.getCurrentPosition(
            resolve, 
            (error) => {
                let errorMessage;
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        errorMessage = 'Location access was denied. Please enable location permissions.';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        errorMessage = 'Location information is unavailable.';
                        break;
                    case error.TIMEOUT:
                        errorMessage = 'Location request timed out.';
                        break;
                    default:
                        errorMessage = 'An unknown error occurred.';
                }
                reject(new Error(errorMessage));
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 60000
            }
        );
    });
}

// Initialize real-time predictions
function initializeRealTimePredictions() {
    // Initial prediction on page load
    scheduleRealTimePrediction();
}

// Schedule real-time prediction with debouncing
function scheduleRealTimePrediction() {
    if (predictionTimeout) {
        clearTimeout(predictionTimeout);
    }

    predictionTimeout = setTimeout(() => {
        getRealTimePrediction();
    }, 500); // 500ms debounce
}

// Get real-time prediction via API
function getRealTimePrediction() {
    const formData = getFormData();

    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    })
        .then(async response => {
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || 'An error occurred');
            }
            return data;
        })
        .then(data => {
            if (data.status === 'error') {
                showError(data.message);
                return;
            }
            updateUIWithPredictions(data);
        })
        .catch(error => {
            console.error('Error getting prediction:', error);
            showError(error.message || 'Failed to get prediction. Please try again.');
        });
}

// Submit form via AJAX
function submitForm() {
    const form = document.getElementById('crop-prediction-form');
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;

    // Show loading state
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    submitBtn.disabled = true;

    const formData = getFormData();

    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            updateUIWithPredictions(data);
            scrollToResults();
        })
        .catch(error => {
            console.error('Error:', error);
            showError('An error occurred while processing your request.');
        })
        .finally(() => {
            // Restore button state
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        });
}

// Get form data as object
function getFormData() {
    return {
        N: parseFloat(document.getElementById('nitrogen').value),
        P: parseFloat(document.getElementById('phosphorus').value),
        K: parseFloat(document.getElementById('potassium').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        ph: parseFloat(document.getElementById('ph').value),
        rainfall: parseFloat(document.getElementById('rainfall').value)
    };
}

// Update UI with prediction results
function updateUIWithPredictions(data) {
    // Reconstruct the data structure the other functions expect
    const primary = {
        crop: data.crop,
        confidence: data.confidence
    };
    const alternatives = data.alternatives; // Use the correct key from the API

    // Combine primary and alternatives for the chart
    const all_predictions = [primary, ...alternatives];

    // --- The rest of the function now works with the corrected data ---

    // Update primary recommendation
    updatePrimaryRecommendation(primary);

    // Update alternative recommendations
    updateAlternativeRecommendations(alternatives);

    // Update chart with top 6 predictions
    updateSuitabilityChart(all_predictions);

    // Show results section
    document.getElementById('results').style.display = 'block';
}

// Update primary recommendation display
function updatePrimaryRecommendation(prediction) {
    const entry = findCropEntry(prediction.crop) || {
        name: prediction.crop,
        season: 'Season information not available',
        water_req: 'Water requirement not available',
        soil_type: 'Soil type not available',
        tips: 'Growing tips not available',
        icon: '🌱'
    };

    const iconContainer = document.getElementById('primary-crop-icon');
    iconContainer.textContent = entry.icon;
    document.getElementById('primary-crop-name').textContent = entry.name;
    document.getElementById('primary-crop-season').textContent = entry.season;
    document.getElementById('primary-crop-water').textContent = entry.water_req;
    document.getElementById('primary-crop-soil').textContent = entry.soil_type;
    document.getElementById('primary-crop-tips').textContent = entry.tips;
    document.getElementById('primary-confidence').textContent = Math.round(prediction.confidence) + '%';
}

// Update alternative recommendations
function updateAlternativeRecommendations(alternatives) {
    const container = document.getElementById('alternative-crops');
    container.innerHTML = '';

    alternatives.forEach(alt => {
        const entry = findCropEntry(alt.crop) || {
            name: alt.crop,
            season: 'Season information not available',
            icon: '🌱'
        };

        const confidence = Math.round(alt.confidence);
        const confidenceClass = getConfidenceClass(confidence);

        const altElement = document.createElement('div');
        altElement.className = 'alternative-crop';
        altElement.innerHTML = `
            <div class="alternative-crop-icon">${entry.icon}</div>
            <h4>${entry.name}</h4>
            <div class="alternative-confidence ${confidenceClass}">${confidence}% Match</div>
            <p>${entry.season}</p>
        `;

        container.appendChild(altElement);
    });
}

// Update suitability chart
function updateSuitabilityChart(topCrops) {
    if (!topCrops || topCrops.length === 0) return;

    const ctx = document.getElementById('suitabilityChart').getContext('2d');

    if (suitabilityChart) {
        suitabilityChart.destroy();
    }

    const labels = topCrops.map(crop => crop.crop);
    const scores = topCrops.map(crop => Math.round(crop.confidence));
    const colors = ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545'];

    suitabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Suitability Score',
                data: scores,
                backgroundColor: colors.slice(0, labels.length),
                borderColor: colors.slice(0, labels.length),
                borderWidth: 1,
                borderRadius: 8,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Crop Suitability Analysis',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Suitability Score (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Crops'
                    }
                }
            }
        }
    });
}

// Get confidence class for styling
function getConfidenceClass(confidence) {
    if (confidence >= 80) return 'confidence-high';
    if (confidence >= 60) return 'confidence-medium';
    return 'confidence-low';
}


// Show error message
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
}

// Scroll to results section
function scrollToResults() {
    const resultsSection = document.getElementById('results');
    if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// Reset form function
function resetForm() {
    document.getElementById('crop-prediction-form').reset();
    document.getElementById('results').style.display = 'none';
    
    // Reset sliders to default values
    const defaultValues = {
        'nitrogen': 50,
        'phosphorus': 40,
        'potassium': 30,
        'temperature': 25,
        'humidity': 60,
        'ph': 6.5,
        'rainfall': 1000  // Changed to better reflect annual rainfall in mm
    };

    Object.keys(defaultValues).forEach(id => {
        const slider = document.getElementById(id);
        const input = document.getElementById(id + '-input');
        if (slider && input) {
            slider.value = defaultValues[id];
            input.value = defaultValues[id];
        }
    });

    // Schedule new prediction
    scheduleRealTimePrediction();
}

// Export data function
function exportData() {
    const formData = getFormData();
    const data = {
        parameters: formData,
        timestamp: new Date().toISOString()
    };
    
    const dataStr = JSON.stringify(data, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = 'crop-analysis-' + new Date().toISOString().split('T')[0] + '.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
}

// Print results function
function printResults() {
    const printContent = document.getElementById('results').innerHTML;
    const originalContent = document.body.innerHTML;
    
    document.body.innerHTML = printContent;
    window.print();
    document.body.innerHTML = originalContent;
    location.reload();
}

// Share results function
function shareResults() {
    const primaryCrop = document.getElementById('primary-crop-name').textContent;
    const confidence = document.getElementById('primary-confidence').textContent;
    
    if (navigator.share) {
        navigator.share({
            title: 'Crop Recommendation',
            text: `FasalAI recommended ${primaryCrop} with ${confidence} confidence!`,
            url: window.location.href
        }).catch(console.error);
    } else {
        // Fallback for browsers that don't support Web Share API
        alert('Share this recommendation: ' + primaryCrop + ' with ' + confidence + ' confidence!');
    }
}

// ---------------- Chatbot -----------------
function setupChatbot() {
    const toggle = document.getElementById('chat-toggle');
    const panel = document.getElementById('chat-panel');
    const closeBtn = document.getElementById('chat-close');
    const form = document.getElementById('chat-form');
    const input = document.getElementById('chat-input');
    const messages = document.getElementById('chat-messages');

    if (!toggle || !panel || !closeBtn || !form || !input || !messages) return;

    function openChat() {
        panel.style.display = 'flex';
        panel.setAttribute('aria-hidden', 'false');
        input.focus();
        if (messages.childElementCount === 0) {
            addBotMessage('Hi! I\'m FasalAI. Ask me about crops, soil, weather, pests, and best practices.');
        }
    }
    function closeChat() {
        panel.style.display = 'none';
        panel.setAttribute('aria-hidden', 'true');
    }

    toggle.addEventListener('click', openChat);
    closeBtn.addEventListener('click', closeChat);

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = input.value.trim();
        if (!text) return;
        addUserMessage(text);
        input.value = '';
        const typing = addBotMessage('<i class="fas fa-ellipsis-h fa-bounce"></i>');
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            typing.remove();
            if (!res.ok || data.status !== 'success') {
                addBotMessage(data.message || 'Sorry, something went wrong.');
                return;
            }
            addBotMessage(data.reply);
        } catch (err) {
            typing.remove();
            addBotMessage('Network error. Please try again.');
        }
    });

    function addUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'chat-msg user';
        div.textContent = text;
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
    }
    function addBotMessage(html) {
        const div = document.createElement('div');
        div.className = 'chat-msg bot';
        div.innerHTML = html;
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
        return div;
    }
}

function scrollToForm() {
    const el = document.getElementById('input-form');
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}