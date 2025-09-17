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

import os
import logging
from typing import Dict, Optional
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def get_live_data(latitude: float, longitude: float) -> Optional[Dict[str, float]]:
    """
    Fetch real-time weather and soil data based on GPS coordinates.
    
    Args:
        latitude: Geographic latitude coordinate
        longitude: Geographic longitude coordinate
        
    Returns:
        Dictionary containing processed weather and soil data with keys:
        - temperature (in °C)
        - humidity (in %)
        - rainfall (in mm)
        - ph (pH value)
        - organic_carbon (in g/kg)
        - N (Nitrogen in cg/kg)
        - P (Phosphorus in mg/kg)
        - K (Potassium in mg/kg)
        
        Returns None if both API calls fail.
    """
    # Initialize result dictionary
    result_data = {}
    
    # Fetch weather data
    weather_data = _fetch_weather_data(latitude, longitude)
    if weather_data:
        result_data.update(weather_data)
        logger.info(f"Weather data fetched successfully: {weather_data}")
    else:
        logger.warning("Failed to fetch weather data")
    
    # Fetch soil data
    soil_data = _fetch_soil_data(latitude, longitude)
    if soil_data:
        result_data.update(soil_data)
        logger.info(f"Soil data fetched successfully: {soil_data}")
    else:
        logger.warning("Failed to fetch soil data")
        # Provide default soil values if API fails
        result_data.update(_get_default_soil_data())
    
    # Return None if no data was successfully fetched
    return result_data if result_data else None


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
            "depths": ",".join(SOILGRIDS_DEPTHS),
            "value": value_method
        }
        
        # Make API request with timeout
        response = requests.get(SOILGRIDS_BASE_URL, params=params, timeout=15)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        soil_json = response.json()
        
        # Extract and process soil properties
        properties_data = soil_json["properties"]
        
        # Process each property with appropriate conversion
        processed_data = {}
        
        # pH (divide by 10.0)
        ph_value = _extract_soil_property(properties_data, "phh2o", "0-5cm") / 10.0
        processed_data["ph"] = round(ph_value, 1)
        
        # Soil Organic Carbon - SOC (divide by 10.0 to get g/kg)
        soc_value = _extract_soil_property(properties_data, "soc", "0-5cm") / 10.0
        processed_data["organic_carbon"] = round(soc_value, 1)
        
        # Nitrogen (divide by 10.0 to get cg/kg)
        n_value = _extract_soil_property(properties_data, "nitrogen", "0-5cm") / 10.0
        processed_data["N"] = round(n_value, 1)
        
        # Phosphorus (divide by 10.0 to get mg/kg)
        p_value = _extract_soil_property(properties_data, "phosphorus", "0-5cm") / 10.0
        processed_data["P"] = round(p_value, 1)
        
        # Potassium (divide by 10.0 to get mg/kg)
        k_value = _extract_soil_property(properties_data, "potassium", "0-5cm") / 10.0
        processed_data["K"] = round(k_value, 1)
        
        return processed_data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Soil API request failed: {e}")
    except KeyError as e:
        logger.error(f"Unexpected soil API response structure: {e}")
    except ValueError as e:
        logger.error(f"Data conversion error in soil response: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in soil data processing: {e}")
    
    return None


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


def _get_default_soil_data() -> Dict[str, float]:
    """
    Provide default soil values when SoilGrids API fails.
    
    Returns:
        Dictionary with default soil values
    """
    return {
        "ph": 6.5,
        "organic_carbon": 15.0,
        "N": 50.0,
        "P": 30.0,
        "K": 150.0
    }


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