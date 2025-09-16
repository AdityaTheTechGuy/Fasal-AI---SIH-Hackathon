// FasalAI - AI Crop Recommendation System JavaScript

// Crop database with detailed information
const cropDatabase = {
    "Rice": { name: "Rice", season: "Kharif (June-November)", water_req: "High (1200-2500mm)", soil_type: "Clay loam, well-drained", tips: "Ensure proper water management and pest control for diseases like blast and brown spot.", icon: "🌾" },
    "Maize": { name: "Maize", season: "Kharif/Rabi (April-July, October-January)", water_req: "Medium (500-800mm)", soil_type: "Well-drained loamy soil", tips: "Apply balanced fertilizers and control stem borer and fall armyworm.", icon: "🌽" },
    "Chickpea": { name: "Chickpea", season: "Rabi (October-April)", water_req: "Low (300-400mm)", soil_type: "Well-drained, neutral pH", tips: "Drought tolerant crop, avoid waterlogging and control pod borer.", icon: "🫘" },
    "Cotton": { name: "Cotton", season: "Kharif (May-November)", water_req: "Medium (700-1200mm)", soil_type: "Deep, well-drained black soil", tips: "Monitor for bollworm attacks and ensure adequate potash supply.", icon: "🌱" },
    "Apple": { name: "Apple", season: "Perennial (Harvest: September-November)", water_req: "Medium (1000-1200mm)", soil_type: "Well-drained, slightly acidic", tips: "Requires cold winters for dormancy. Prune regularly and control scab disease.", icon: "🍎" },
    "Banana": { name: "Banana", season: "Year-round planting", water_req: "High (1500-1800mm)", soil_type: "Rich, well-drained loamy soil", tips: "High potassium requirement. Protect from strong winds and control nematodes.", icon: "🍌" },
    "Coffee": { name: "Coffee", season: "Perennial (Harvest: December-February)", water_req: "Medium (1500-2000mm)", soil_type: "Well-drained, slightly acidic", tips: "Shade-grown crop. Control coffee berry borer and leaf rust disease.", icon: "☕" },
    "Kidneybeans": { name: "Kidneybeans", season: "Rabi (October-March)", water_req: "Medium (400-500mm)", soil_type: "Well-drained loamy soil", tips: "Avoid waterlogging and ensure proper nitrogen fixation.", icon: "🫘" },
    "Pigeonpeas": { name: "Pigeonpeas", season: "Kharif (June-December)", water_req: "Medium (600-650mm)", soil_type: "Well-drained sandy loam", tips: "Drought tolerant, good for intercropping systems.", icon: "🫛" },
    "Mothbeans": { name: "Mothbeans", season: "Kharif (July-October)", water_req: "Low (300-400mm)", soil_type: "Sandy, drought-prone areas", tips: "Highly drought tolerant, suitable for arid regions.", icon: "🫘" },
    "Mungbean": { name: "Mungbean", season: "Kharif/Summer (March-June, July-October)", water_req: "Medium (400-500mm)", soil_type: "Well-drained sandy loam", tips: "Short duration crop, good for crop rotation.", icon: "🫛" },
    "Blackgram": { name: "Blackgram", season: "Kharif/Rabi (June-September, October-January)", water_req: "Medium (400-500mm)", soil_type: "Well-drained loamy soil", tips: "Sensitive to waterlogging, ensure good drainage.", icon: "🫘" },
    "Lentil": { name: "Lentil", season: "Rabi (October-April)", water_req: "Low (300-400mm)", soil_type: "Well-drained sandy loam", tips: "Cold tolerant, avoid excessive moisture during flowering.", icon: "🫛" },
    "Pomegranate": { name: "Pomegranate", season: "Year-round (Peak: October-February)", water_req: "Medium (500-700mm)", soil_type: "Well-drained, slightly alkaline", tips: "Drought tolerant once established, prune regularly.", icon: "🍎" },
    "Mango": { name: "Mango", season: "Perennial (Harvest: April-July)", water_req: "Medium (750-1200mm)", soil_type: "Well-drained, deep soil", tips: "Avoid waterlogging during flowering, control fruit fly.", icon: "🥭" },
    "Grapes": { name: "Grapes", season: "Perennial (Harvest: February-April)", water_req: "Medium (500-700mm)", soil_type: "Well-drained, slightly alkaline", tips: "Requires pruning and trellising, control powdery mildew.", icon: "🍇" },
    "Watermelon": { name: "Watermelon", season: "Summer (February-May)", water_req: "Medium (400-600mm)", soil_type: "Sandy loam, well-drained", tips: "Requires warm weather, control fruit fly and aphids.", icon: "🍉" },
    "Muskmelon": { name: "Muskmelon", season: "Summer (February-May)", water_req: "Medium (300-500mm)", soil_type: "Sandy loam, well-drained", tips: "Warm season crop, ensure adequate potash supply.", icon: "🍈" },
    "Orange": { name: "Orange", season: "Perennial (Harvest: December-February)", water_req: 'Medium (1000-1200mm)', soil_type: "Well-drained, slightly acidic", tips: "Regular irrigation needed, control citrus canker.", icon: "🍊" },
    "Papaya": { name: "Papaya", season: "Year-round planting", water_req: "High (1200-1500mm)", soil_type: "Well-drained, rich organic matter", tips: "Avoid waterlogging, control papaya ring spot virus.", icon: "🍈" },
    "Coconut": { name: "Coconut", season: "Year-round planting", water_req: "High (1500-2000mm)", soil_type: "Coastal sandy soil", tips: "High potash requirement, control rhinoceros beetle.", icon: "🥥" },
    "Jute": { name: "Jute", season: "Kharif (April-August)", water_req: "High (1000-1500mm)", soil_type: "Alluvial soil with high moisture", tips: "Requires high humidity, harvest at proper maturity.", icon: "🌾" }
};

let suitabilityChart = null;
let predictionTimeout = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function () {
    initializeSliders();
    setupEventListeners();
    initializeRealTimePredictions();
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
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                console.error('Prediction error:', data.error);
                return;
            }
            updateUIWithPredictions(data);
        })
        .catch(error => {
            console.error('Error getting prediction:', error);
        });
}

// Submit form via AJAX
function submitForm() {
    const form = document.getElementById('crop-prediction-form');
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;

    // Show loading state
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> {{ _("Analyzing...") }}';
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
    const primary = data.primary_prediction;
    const alternatives = data.alternative_predictions;

    // Update primary recommendation
    updatePrimaryRecommendation(primary);

    // Update alternative recommendations
    updateAlternativeRecommendations(alternatives);

    // Update chart with top 6 predictions
    updateSuitabilityChart(data.all_predictions.slice(0, 6));

    // Show results section
    document.getElementById('results').style.display = 'block';
}

// Update primary recommendation display
function updatePrimaryRecommendation(prediction) {
    const crop = cropDatabase[prediction.crop] || {
        name: prediction.crop,
        season: 'Season information not available',
        water_req: 'Water requirement not available',
        soil_type: 'Soil type not available',
        tips: 'Growing tips not available',
        icon: '🌱'
    };

    document.getElementById('primary-crop-icon').textContent = crop.icon;
    document.getElementById('primary-crop-name').textContent = crop.name;
    document.getElementById('primary-crop-season').textContent = crop.season;
    document.getElementById('primary-crop-water').textContent = crop.water_req;
    document.getElementById('primary-crop-soil').textContent = crop.soil_type;
    document.getElementById('primary-crop-tips').textContent = crop.tips;
    document.getElementById('primary-confidence').textContent = Math.round(prediction.confidence) + '%';
}

// Update alternative recommendations
function updateAlternativeRecommendations(alternatives) {
    const container = document.getElementById('alternative-crops');
    container.innerHTML = '';

    alternatives.forEach(alt => {
        const crop = cropDatabase[alt.crop] || {
            name: alt.crop,
            season: 'Season information not available',
            icon: '🌱'
        };

        const confidence = Math.round(alt.confidence);
        const confidenceClass = getConfidenceClass(confidence);

        const altElement = document.createElement('div');
        altElement.className = 'alternative-crop';
        altElement.innerHTML = `
            <div class="alternative-crop-icon">${crop.icon}</div>
            <h4>${crop.name}</h4>
            <div class="alternative-confidence ${confidenceClass}">${confidence}% Match</div>
            <p>${crop.season}</p>
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
                    },
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Crops'
                    },
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// Show error message
function showError(message) {
    // You can implement a proper error display here
    console.error('Error:', message);
    alert('Error: ' + message);
}

// Scroll to results section
function scrollToResults() {
    document.getElementById('results').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// Scroll to form function
function scrollToForm() {
    document.getElementById('input-form').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// Text-to-speech function
function playText(elementId) {
    const text = document.getElementById(elementId).textContent;
    const lang = document.documentElement.lang;
    window.open(`/speak/${encodeURIComponent(text)}`, '_blank');
}

// Utility function to get confidence class
function getConfidenceClass(confidence) {
    if (confidence >= 80) return 'confidence-high';
    if (confidence >= 60) return 'confidence-medium';
    return 'confidence-low';
}

// Add some interactive enhancements
document.addEventListener('DOMContentLoaded', function () {
    // Add smooth hover effects to cards
    const cards = document.querySelectorAll('.card, .parameter-card, .stat-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function () {
            this.style.transform = 'translateY(-2px)';
        });

        card.addEventListener('mouseleave', function () {
            this.style.transform = 'translateY(0)';
        });
    });
});

// Export functions for potential testing
window.FasalAI = {
    scrollToForm,
    playText,
    getRealTimePrediction
};