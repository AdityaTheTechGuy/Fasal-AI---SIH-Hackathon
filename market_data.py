import requests
from datetime import datetime

def get_crop_market_data(crop_name):
    # Sample market data (in a real system, this would come from an API)
    market_data = {
        'Rice': {'price_range': '2000-2500', 'yield_per_hectare': '3.5-4.0', 'unit': 'quintal'},
        'Maize': {'price_range': '1800-2200', 'yield_per_hectare': '2.5-3.5', 'unit': 'quintal'},
        'Chickpea': {'price_range': '4500-5000', 'yield_per_hectare': '1.5-2.0', 'unit': 'quintal'},
        'Cotton': {'price_range': '5500-6000', 'yield_per_hectare': '20-25', 'unit': 'quintal'},
        'Apple': {'price_range': '5000-7000', 'yield_per_hectare': '15-20', 'unit': 'quintal'},
        'Banana': {'price_range': '2500-3000', 'yield_per_hectare': '350-400', 'unit': 'quintal'},
        'Coffee': {'price_range': '15000-18000', 'yield_per_hectare': '1.5-2.0', 'unit': 'quintal'},
        'Kidneybeans': {'price_range': '6000-7000', 'yield_per_hectare': '12-15', 'unit': 'quintal'},
        'Pigeonpeas': {'price_range': '5500-6500', 'yield_per_hectare': '15-18', 'unit': 'quintal'},
        'Mothbeans': {'price_range': '4500-5500', 'yield_per_hectare': '8-10', 'unit': 'quintal'},
        'Mungbean': {'price_range': '7000-8000', 'yield_per_hectare': '8-12', 'unit': 'quintal'},
        'Blackgram': {'price_range': '6000-7000', 'yield_per_hectare': '8-10', 'unit': 'quintal'},
        'Lentil': {'price_range': '5500-6500', 'yield_per_hectare': '10-12', 'unit': 'quintal'},
        'Pomegranate': {'price_range': '8000-10000', 'yield_per_hectare': '150-200', 'unit': 'quintal'},
        'Mango': {'price_range': '4000-6000', 'yield_per_hectare': '100-150', 'unit': 'quintal'},
        'Grapes': {'price_range': '5000-7000', 'yield_per_hectare': '250-300', 'unit': 'quintal'},
    }
    
    return market_data.get(crop_name, {
        'price_range': 'Data not available',
        'yield_per_hectare': 'Data not available',
        'unit': 'quintal'
    })