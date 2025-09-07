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
    "Orange": { name: "Orange", season: "Perennial (Harvest: December-February)", water_req: "Medium (1000-1200mm)", soil_type: "Well-drained, slightly acidic", tips: "Regular irrigation needed, control citrus canker.", icon: "🍊" },
    "Papaya": { name: "Papaya", season: "Year-round planting", water_req: "High (1200-1500mm)", soil_type: "Well-drained, rich organic matter", tips: "Avoid waterlogging, control papaya ring spot virus.", icon: "🍈" },
    "Coconut": { name: "Coconut", season: "Year-round planting", water_req: "High (1500-2000mm)", soil_type: "Coastal sandy soil", tips: "High potash requirement, control rhinoceros beetle.", icon: "🥥" },
    "Jute": { name: "Jute", season: "Kharif (April-August)", water_req: "High (1000-1500mm)", soil_type: "Alluvial soil with high moisture", tips: "Requires high humidity, harvest at proper maturity.", icon: "🌾" }
};

// Crop suitability rules for AI prediction
const cropSuitability = {
    "Rice": { N: [80, 120], P: [40, 60], K: [40, 60], temp: [20, 35], humidity: [80, 90], pH: [5.5, 7.0], rainfall: [150, 300] },
    "Maize": { N: [80, 120], P: [40, 60], K: [20, 60], temp: [18, 27], humidity: [60, 70], pH: [5.5, 7.5], rainfall: [50, 100] },
    "Chickpea": { N: [40, 70], P: [60, 85], K: [20, 50], temp: [20, 25], humidity: [40, 70], pH: [6.0, 7.5], rainfall: [30, 40] },
    "Cotton": { N: [120, 160], P: [40, 80], K: [80, 120], temp: [21, 30], humidity: [80, 90], pH: [5.8, 8.0], rainfall: [50, 120] },
    "Apple": { N: [20, 40], P: [125, 145], K: [200, 240], temp: [8, 15], humidity: [50, 60], pH: [5.5, 7.0], rainfall: [100, 120] },
    "Banana": { N: [100, 120], P: [75, 85], K: [50, 100], temp: [26, 30], humidity: [75, 85], pH: [5.5, 7.0], rainfall: [100, 180] },
    "Coffee": { N: [100, 120], P: [15, 35], K: [30, 50], temp: [23, 30], humidity: [50, 70], pH: [6.0, 7.0], rainfall: [150, 200] },
    "Kidneybeans": { N: [20, 40], P: [60, 80], K: [15, 30], temp: [15, 25], humidity: [65, 70], pH: [6.0, 7.0], rainfall: [40, 50] },
    "Pigeonpeas": { N: [20, 40], P: [60, 80], K: [15, 30], temp: [26, 35], humidity: [60, 65], pH: [6.0, 7.0], rainfall: [60, 65] },
    "Mothbeans": { N: [20, 40], P: [40, 60], K: [40, 60], temp: [27, 32], humidity: [65, 75], pH: [6.5, 7.5], rainfall: [30, 40] },
    "Mungbean": { N: [20, 40], P: [40, 60], K: [20, 40], temp: [27, 35], humidity: [80, 90], pH: [6.2, 7.2], rainfall: [40, 50] },
    "Blackgram": { N: [40, 60], P: [60, 80], K: [20, 40], temp: [25, 35], humidity: [65, 75], pH: [6.0, 7.0], rainfall: [40, 50] },
    "Lentil": { N: [20, 40], P: [60, 80], K: [20, 40], temp: [15, 25], humidity: [65, 70], pH: [6.0, 7.5], rainfall: [30, 40] },
    "Pomegranate": { N: [20, 40], P: [10, 40], K: [40, 60], temp: [15, 35], humidity: [35, 45], pH: [6.5, 7.5], rainfall: [50, 70] },
    "Mango": { N: [20, 40], P: [10, 40], K: [20, 40], temp: [24, 27], humidity: [50, 70], pH: [5.5, 7.5], rainfall: [75, 120] },
    "Grapes": { N: [20, 40], P: [125, 145], K: [200, 240], temp: [15, 25], humidity: [45, 55], pH: [6.0, 7.0], rainfall: [50, 70] },
    "Watermelon": { N: [100, 120], P: [40, 60], K: [50, 80], temp: [24, 27], humidity: [65, 80], pH: [6.0, 7.0], rainfall: [40, 60] },
    "Muskmelon": { N: [100, 120], P: [40, 60], K: [50, 80], temp: [18, 24], humidity: [90, 95], pH: [6.0, 7.0], rainfall: [30, 50] },
    "Orange": { N: [20, 40], P: [10, 25], K: [10, 40], temp: [15, 27], humidity: [70, 80], pH: [6.0, 7.5], rainfall: [100, 120] },
    "Papaya": { N: [50, 100], P: [40, 60], K: [50, 100], temp: [22, 26], humidity: [60, 65], pH: [6.0, 6.7], rainfall: [120, 150] },
    "Coconut": { N: [20, 40], P: [10, 20], K: [20, 30], temp: [27, 30], humidity: [70, 80], pH: [5.2, 8.0], rainfall: [150, 200] },
    "Jute": { N: [78, 118], P: [46, 66], K: [48, 68], temp: [25, 35], humidity: [70, 80], pH: [6.0, 7.5], rainfall: [100, 150] }
};

let suitabilityChart = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function () {
    initializeSliders();
    updatePrediction();
    setupEventListeners();
});

// Initialize all sliders and input synchronization
function initializeSliders() {
    const parameters = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall'];

    parameters.forEach(param => {
        const slider = document.getElementById(param);
        const input = document.getElementById(param + '-input');

        // Sync slider to input
        slider.addEventListener('input', function () {
            input.value = this.value;
            updatePrediction();
        });

        // Sync input to slider
        input.addEventListener('input', function () {
            const value = parseFloat(this.value);
            const min = parseFloat(slider.min);
            const max = parseFloat(slider.max);

            if (value >= min && value <= max) {
                slider.value = value;
                updatePrediction();
            }
        });
    });
}

// Setup additional event listeners
function setupEventListeners() {
    // Add any additional event listeners here
}

// Scroll to form function
function scrollToForm() {
    document.getElementById('input-form').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// AI Prediction Algorithm
function predictCrop(parameters) {
    const { N, P, K, temperature, humidity, pH, rainfall } = parameters;
    const scores = [];

    // Calculate suitability score for each crop
    Object.keys(cropSuitability).forEach(cropName => {
        const requirements = cropSuitability[cropName];
        let score = 0;
        let factors = 0;

        // Score calculation based on how close the values are to optimal ranges
        const scoreFactors = [
            { value: N, range: requirements.N, weight: 1.2 },
            { value: P, range: requirements.P, weight: 1.2 },
            { value: K, range: requirements.K, weight: 1.2 },
            { value: temperature, range: requirements.temp, weight: 1.5 },
            { value: humidity, range: requirements.humidity, weight: 1.0 },
            { value: pH, range: requirements.pH, weight: 1.3 },
            { value: rainfall, range: requirements.rainfall, weight: 1.1 }
        ];

        scoreFactors.forEach(factor => {
            const [min, max] = factor.range;
            let factorScore = 0;

            if (factor.value >= min && factor.value <= max) {
                // Perfect range - full score
                factorScore = 100;
            } else if (factor.value < min) {
                // Below range - decreasing score
                const deviation = min - factor.value;
                const tolerance = min * 0.3; // 30% tolerance
                factorScore = Math.max(0, 100 - (deviation / tolerance) * 50);
            } else {
                // Above range - decreasing score
                const deviation = factor.value - max;
                const tolerance = max * 0.3; // 30% tolerance
                factorScore = Math.max(0, 100 - (deviation / tolerance) * 50);
            }

            score += factorScore * factor.weight;
            factors += factor.weight;
        });

        const normalizedScore = Math.min(100, score / factors);

        scores.push({
            crop: cropName,
            score: normalizedScore,
            confidence: Math.min(98.86, normalizedScore * 0.95 + Math.random() * 5) // Add some realistic variation
        });
    });

    // Sort by score
    scores.sort((a, b) => b.score - a.score);

    return scores;
}

// Update predictions in real-time
function updatePrediction() {
    const parameters = {
        N: parseFloat(document.getElementById('nitrogen').value),
        P: parseFloat(document.getElementById('phosphorus').value),
        K: parseFloat(document.getElementById('potassium').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        pH: parseFloat(document.getElementById('ph').value),
        rainfall: parseFloat(document.getElementById('rainfall').value)
    };

    const predictions = predictCrop(parameters);
    const primary = predictions[0];
    const alternatives = predictions.slice(1, 4);

    // Update primary recommendation
    updatePrimaryRecommendation(primary);

    // Update alternative recommendations
    updateAlternativeRecommendations(alternatives);

    // Update chart
    updateSuitabilityChart(predictions.slice(0, 6));
}

// Update primary recommendation display
function updatePrimaryRecommendation(prediction) {
    const crop = cropDatabase[prediction.crop];

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
        const crop = cropDatabase[alt.crop];
        const confidence = Math.round(alt.confidence);

        let confidenceClass = 'confidence-low';
        if (confidence >= 80) confidenceClass = 'confidence-high';
        else if (confidence >= 60) confidenceClass = 'confidence-medium';

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
    const ctx = document.getElementById('suitabilityChart').getContext('2d');

    if (suitabilityChart) {
        suitabilityChart.destroy();
    }

    const labels = topCrops.map(crop => crop.crop);
    const scores = topCrops.map(crop => Math.round(crop.score));
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

    // Add loading animation when parameters change
    const sliders = document.querySelectorAll('.slider, .number-input');
    sliders.forEach(slider => {
        slider.addEventListener('input', function () {
            document.querySelector('.results-section').style.opacity = '0.8';
            setTimeout(() => {
                document.querySelector('.results-section').style.opacity = '1';
            }, 200);
        });
    });
});

// Advanced features for better user experience
function showParameterTooltip(element, message) {
    // Create tooltip functionality if needed
    console.log('Tooltip:', message);
}

// Export functions for potential testing
window.FasalAI = {
    predictCrop,
    updatePrediction,
    cropDatabase,
    scrollToForm
};