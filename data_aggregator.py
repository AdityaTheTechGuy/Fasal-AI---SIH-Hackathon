"""
Data Aggregation Module for Fasal AI Crop Recommendation System

This module fetches real-time weather and soil data from external APIs based on
GPS coordinates and returns processed data for crop prediction.

API Sources:
- OpenWeatherMap: Current weather data
- SoilGrids: Estimated soil properties

Author: Fasal AI Team
Date: 2025
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import logging
from typing import Dict, Optional
import requests


OWM_KEY = os.getenv("OWM_API_KEY")
SESSION = requests.Session()
TIMEOUT = (5, 12)

def _req(url, **kw):
    r = SESSION.get(url, timeout=TIMEOUT, **kw)
    r.raise_for_status()
    return r.json()

def fetch_weather_now(lat, lon):
    """Temperature & humidity (rain if present)."""
    if not OWM_KEY:
        return {}
    url = "https://api.openweathermap.org/data/2.5/weather"
    j = _req(url, params={"lat": lat, "lon": lon, "units": "metric", "appid": OWM_KEY})
    main = j.get("main") or {}
    rain_now = (j.get("rain") or {})
    return {
        "temperature": main.get("temp"),
        "humidity": main.get("humidity"),
        "rain_now": rain_now.get("1h") or rain_now.get("3h")
    }

def fetch_annual_rainfall(lat, lon):
    """Calculate annual rainfall using historical monthly data."""
    if not OWM_KEY:
        return None
    try:
        # Use history API to get monthly rainfall data
        url = "https://history.openweathermap.org/data/2.5/aggregated/year"
        j = _req(url, params={
            "lat": lat, "lon": lon, 
            "appid": OWM_KEY
        })
        
        # Calculate total annual rainfall from monthly data
        annual_rainfall = 0.0
        if "result" in j:
            for month in j["result"]:
                # Get monthly precipitation in mm
                monthly_rain = month.get("precipitation", {}).get("mean", 0.0)
                annual_rainfall += float(monthly_rain * 30)  # approximate days per month
                
        # Fallback: If historical data unavailable, estimate from recent data
        if annual_rainfall == 0:
            # Get 5 day forecast which includes historical data
            url = "https://api.openweathermap.org/data/2.5/forecast"
            j = _req(url, params={
                "lat": lat, "lon": lon, 
                "units": "metric",
                "appid": OWM_KEY
            })
            
            # Calculate average daily rainfall from 5-day data
            daily_rain = 0.0
            rain_days = 0
            for item in j.get("list", []):
                rain = item.get("rain", {}).get("3h", 0.0)
                if rain > 0:
                    daily_rain += rain
                    rain_days += 1
            
            if rain_days > 0:
                # Extrapolate to annual based on average daily rainfall
                avg_daily = daily_rain / rain_days
                annual_rainfall = avg_daily * 365
                log_rainfall_data(lat, lon, annual_rainfall, "estimated from 5-day forecast")
            
        if annual_rainfall > 0:
            log_rainfall_data(lat, lon, annual_rainfall, "historical monthly data")
            
        return annual_rainfall
        
    except Exception as e:
        logger.error(f"Error fetching annual rainfall: {e}")
        return None

def fetch_soil(lat, lon):
    """SoilGrids: N, P, K, pH at 0–5 cm depth."""
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    j = _req(url, params={
        "lat": lat, "lon": lon,
        "depth": "0-5cm",
        "value": "mean",
        "property": "phh2o,nitrogen,phosphorus,potassium"
    })
    props = j.get("properties") or {}
    def pick(prop):
        p = props.get(prop) or {}
        if "mean" in p: return p["mean"]
        if "values" in p and p["values"]:
            return p["values"][0].get("value")
        return None
    return {
        "ph": pick("phh2o"),
        "N":  pick("nitrogen"),
        "P":  pick("phosphorus"),
        "K":  pick("potassium"),
    }


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_rainfall_data(lat, lon, rainfall, source):
    """
    Log rainfall data collection for monitoring and debugging
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        rainfall: Rainfall value in mm
        source: Source of the rainfall data (historical/estimated)
    """
    logger.info(f"Rainfall data collected for ({lat}, {lon}): {rainfall}mm from {source}")

# API Configuration - Keys should be set as environment variables
OWM_API_KEY = os.getenv("OWM_API_KEY")
if not OWM_API_KEY:
    logger.warning("OpenWeatherMap API key not found in environment variables")

SOILGRIDS_BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# SoilGrids property mapping
SOILGRIDS_PROPERTIES = {
    "phh2o": "ph",
    "soc": "organic_carbon",
    "nitrogen": "N",
    "phosphorus": "P", 
    "potassium": "K"
}

# SoilGrids depth layers
SOILGRIDS_DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]

def get_live_data(lat, lon):
    """
    Fetch live soil + weather data for the given coordinates.
    Returns dict with keys: N, P, K, ph, temperature, humidity, rainfall.
    Uses fallback values and district data when needed.
    """
    result = {}
    try:
        # --- Weather ---
        wx_now = fetch_weather_now(lat, lon) or {}
        annual_rain = fetch_annual_rainfall(lat, lon)
        
        result.update({
            "temperature": wx_now.get("temperature", 25.0),  # Reasonable default
            "humidity": wx_now.get("humidity", 60.0),       # Reasonable default
            "rainfall": annual_rain or 1200.0               # Average annual rainfall
        })

        # --- Soil (try SoilGrids first) ---
        try:
            soil = fetch_soil(lat, lon) or {}
            if any(soil.get(key) is not None for key in ['N', 'P', 'K', 'ph']):
                result.update({
                    "N": soil.get("N", 50.0),    # Default moderate nitrogen
                    "P": soil.get("P", 50.0),    # Default moderate phosphorus  
                    "K": soil.get("K", 35.0),    # Default moderate potassium
                    "ph": soil.get("ph", 6.5),   # Default neutral pH
                    "soil_source": "soilgrids"
                })
            else:
                # SoilGrids returned empty data, try detailed soil data fetch
                soil_data = _fetch_soil_data(lat, lon)
                if soil_data:
                    result.update(soil_data)
                    result["soil_source"] = "detailed"
        except Exception as soil_err:
            logger.warning(f"SoilGrids fetch failed: {soil_err}")

        # If still missing soil data, try district fallback
        if not all(key in result for key in ['N', 'P', 'K', 'ph']):
            # This would come from reverse geocoding or user input
            district_npk = get_district_npk("Adalahatu")  # Using first district as fallback
            if district_npk:
                result.update(district_npk)
                result["soil_source"] = "district"
            else:
                # Final fallback - use reasonable defaults
                result.update({
                    "N": 50.0,    # Moderate nitrogen
                    "P": 50.0,    # Moderate phosphorus
                    "K": 35.0,    # Moderate potassium
                    "ph": 6.5,    # Neutral pH
                    "soil_source": "default"
                })

        # Ensure all required fields are present with numeric values
        for key in ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']:
            if key not in result or result[key] is None:
                result[key] = {
                    'N': 50.0, 'P': 50.0, 'K': 35.0, 'ph': 6.5,
                    'temperature': 25.0, 'humidity': 60.0, 'rainfall': 1200.0
                }[key]

        return result

    except Exception as e:
        logger.exception("Error in get_live_data")
        # Emergency fallback - guarantee we return something usable
        return {
            "N": 50.0,
            "P": 50.0,
            "K": 35.0,
            "ph": 6.5,
            "temperature": 25.0,
            "humidity": 60.0,
            "rainfall": 1200.0,
            "soil_source": "emergency_fallback"
        }

def _fetch_weather_data(latitude: float, longitude: float) -> Optional[Dict[str, float]]:
    """
    Fetch current weather data from OpenWeatherMap API.
    
    Args:
        latitude: Geographic latitude coordinate
        longitude: Geographic longitude coordinate
        
    Returns:
        Dictionary with weather data or None if request fails
    """
    if not OWM_API_KEY:
        logger.error("OpenWeatherMap API key not configured")
        return None
    
    try:
        # Construct API request URL
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": OWM_API_KEY,
            "units": "metric"  # Get temperature in Celsius
        }
        
        # Make API request with timeout
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        weather_json = response.json()
        
        # Extract required parameters with error handling
        temperature = weather_json["main"]["temp"]
        humidity = weather_json["main"]["humidity"]
        
        # Handle rainfall data (key may not exist if no rain)
        rainfall = weather_json.get("rain", {}).get("1h", 0.0)
        # If no hourly rain, check for daily rain
        if rainfall == 0.0:
            rainfall = weather_json.get("rain", {}).get("3h", 0.0)
        
        return {
            "temperature": float(temperature),
            "humidity": float(humidity),
            "rainfall": float(rainfall)
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API request failed: {e}")
    except KeyError as e:
        logger.error(f"Unexpected weather API response structure: {e}")
    except ValueError as e:
        logger.error(f"Data conversion error in weather response: {e}")
    
    return None


def get_district_npk(district_name):
    """Fallback soil data from district records."""
    try:
        import pandas as pd
        district_data = pd.read_csv("jh_district_npk.csv")
        name_norm = str(district_name).strip().lower()
        district_data['district_norm'] = district_data['district'].str.strip().str.lower()
        row = district_data[district_data['district_norm'] == name_norm]
        if not row.empty:
            return {
                "N": float(row.iloc[0]["N"]),
                "P": float(row.iloc[0]["P"]),
                "K": float(row.iloc[0]["K"]),
                "ph": float(row.iloc[0]["ph"])
            }
    except Exception:
        pass
    return None

def _fetch_soil_data(latitude: float, longitude: float) -> Optional[Dict[str, float]]:
    """
    Fetch soil data from SoilGrids API.
    
    Args:
        latitude: Geographic latitude coordinate
        longitude: Geographic longitude coordinate
        
    Returns:
        Dictionary with processed soil data or None if request fails
    """
    try:
        # Define soil properties to query
        properties = list(SOILGRIDS_PROPERTIES.keys())
        value_method = "mean"  # Use mean value for the depth
        
        # Construct API request
        params = {
            "lon": longitude,
            "lat": latitude,
            "properties": ",".join(properties),
            "depths": SOILGRIDS_DEPTHS[0],  # Just use top layer for now
            "value": value_method
        }
        
        # Make API request with timeout
        response = requests.get(SOILGRIDS_BASE_URL, params=params, timeout=15)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        soil_json = response.json()
        
        # Extract and process soil properties
        properties_data = soil_json.get("properties", {})
        
        # Get fallback values from district data if possible
        fallback_values = {}
        if 'district' in soil_json:
            district_name = soil_json['district']
            fallback_values = get_district_npk(district_name) or {}
        
        # Process each property with appropriate conversion and fallbacks
        processed_data = {}
        
        # Use default ranges if properties not found
        default_ranges = {
            "N": (30, 80),  # Typical nitrogen range
            "P": (40, 80),  # Typical phosphorus range
            "K": (25, 50),  # Typical potassium range
            "ph": (6.0, 7.0)  # Typical pH range
        }
        
        # Helper to get a reasonable default
        def get_default(key):
            if key in fallback_values:
                return fallback_values[key]
            min_val, max_val = default_ranges[key]
            return (min_val + max_val) / 2
            
        try:
            # pH (divide by 10.0)
            ph_data = properties_data.get("phh2o", {}).get("values", [{}])[0]
            ph_value = float(ph_data.get("value", get_default("ph") * 10.0)) / 10.0
            processed_data["ph"] = round(ph_value, 1)
            
            # Nitrogen (divide by 10.0 to get cg/kg)
            n_data = properties_data.get("nitrogen", {}).get("values", [{}])[0]
            n_value = float(n_data.get("value", get_default("N") * 10.0)) / 10.0
            processed_data["N"] = round(n_value, 1)
            
            # Phosphorus (divide by 10.0 to get mg/kg)
            p_data = properties_data.get("phosphorus", {}).get("values", [{}])[0]
            p_value = float(p_data.get("value", get_default("P") * 10.0)) / 10.0
            processed_data["P"] = round(p_value, 1)
            
            # Potassium (divide by 10.0 to get mg/kg)
            k_data = properties_data.get("potassium", {}).get("values", [{}])[0]
            k_value = float(k_data.get("value", get_default("K") * 10.0)) / 10.0
            processed_data["K"] = round(k_value, 1)
            
            return processed_data
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error processing soil data, using defaults: {e}")
            # Use all defaults if parsing fails
            return {
                "ph": get_default("ph"),
                "N": get_default("N"),
                "P": get_default("P"),
                "K": get_default("K")
            }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Soil API request failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in soil data processing: {e}")
    
    # If all else fails, return typical values
    return {
        "ph": 6.5,  # Neutral pH
        "N": 50,   # Moderate nitrogen
        "P": 50,   # Moderate phosphorus
        "K": 35    # Moderate potassium
    }


def _extract_soil_property(properties_data: Dict, property_name: str, depth: str = "0-5cm") -> float:
    """
    Extract a specific property value from SoilGrids API response.
    
    Args:
        properties_data: The properties section of SoilGrids API response
        property_name: Name of the property to extract
        depth: Depth layer to extract from (default: "0-5cm")
        
    Returns:
        Extracted property value as float
        
    Raises:
        KeyError: If the property or depth is not found in the response
        ValueError: If the value cannot be converted to float
    """
    for layer in properties_data["layers"]:
        if layer["name"] == property_name:
            # Find the specified depth
            for depth_data in layer["depths"]:
                if depth_data["label"] == depth:
                    values = depth_data["values"]
                    return float(values["mean"])
    
    raise KeyError(f"Property {property_name} or depth {depth} not found in API response")


def validate_coordinates(latitude: float, longitude: float) -> bool:
    """
    Validate geographic coordinates.
    
    Args:
        latitude: Latitude coordinate (-90 to 90)
        longitude: Longitude coordinate (-180 to 180)
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    return (-90 <= latitude <= 90) and (-180 <= longitude <= 180)


# Example usage and testing
if __name__ == "__main__":
    # Test with example coordinates (New Delhi, India)
    test_lat = 28.6139
    test_lon = 77.2090
    
    # Set API key for testing (in production, set as environment variable)
    os.environ["OWM_API_KEY"] = "your_openweathermap_api_key_here"
    
    print(f"Testing with coordinates: {test_lat}, {test_lon}")
    
    if validate_coordinates(test_lat, test_lon):
        data = get_live_data(test_lat, test_lon)
        
        if data:
            print("Successfully fetched live data:")
            for key, value in data.items():
                print(f"{key}: {value}")
        else:
            print("Failed to fetch data from both APIs")
    else:
        print("Invalid coordinates provided")