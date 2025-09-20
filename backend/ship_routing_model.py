import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import requests
from datetime import datetime, timedelta
import pickle
import math
import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings
from geopy.distance import geodesic
import random
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.prepared import prep

warnings.filterwarnings('ignore')

@dataclass
class WeatherData:
    """Weather data structure"""
    lat: float
    lon: float
    wind_speed: float
    wind_direction: float
    wave_height: float
    wave_period: float
    wave_direction: float
    temperature: float
    visibility: float
    timestamp: datetime

@dataclass
class RoutePoint:
    """Route point with weather conditions"""
    lat: float
    lon: float
    weather: WeatherData
    fuel_cost: float
    time_cost: float
    risk_factor: float

class LandMaskService:
    """
    Service to detect land vs sea areas using accurate geospatial data.
    This version creates a specific sea polygon and uses morphological closing
    to prevent navigation up wide rivers and estuaries.
    """
    def __init__(self, filepath="land_polygons.pkl"):
        """Initializes the service by creating a navigable sea area polygon."""
        try:
            with open(filepath, 'rb') as f:
                land_gdf = pickle.load(f)
            
            # Create a single unified polygon of all landmasses.
            all_land_union = land_gdf.unary_union

            # --- FIX: Morphological Closing to fill river mouths ---
            # Apply a positive buffer then a negative buffer. This closes gaps
            # in the land polygon, effectively treating large river estuaries as land.
            # The buffer distance is in degrees. 0.05 degrees is roughly 5.5 km.
            closing_distance = 0.05
            closed_land_polygon = all_land_union.buffer(closing_distance).buffer(-closing_distance)
            print("ℹ️  Applied morphological closing to land mask to fill rivers.")

            # Define a large bounding box for the navigable ocean area.
            ocean_box = box(30, -40, 130, 40)

            # Subtract the 'closed' land polygon from the ocean box. This now
            # correctly creates a sea area that excludes inland waterways.
            sea_polygon = ocean_box.difference(closed_land_polygon)
            
            # Prepare the final sea area geometry for fast lookups.
            self.sea_area = prep(sea_polygon)
            print("✅ Navigable sea area mask created successfully (excluding rivers and land).")
        except FileNotFoundError:
            print(f"❌ ERROR: Land mask file not found at '{filepath}'.")
            print("🚢 Land avoidance will be disabled. Routes may cross land.")
            self.sea_area = None
        except Exception as e:
            print(f"❌ ERROR: Could not create sea area mask: {e}")
            self.sea_area = None

    def is_on_land(self, lat: float, lon: float) -> bool:
        """
        Checks if a coordinate is outside the defined navigable sea area.
        Returns True for land, rivers, and any non-sea areas.
        """
        if not self.sea_area:
            return False  # Fail safe: if mask failed, don't block anything
        point = Point(lon, lat)
        # If the point is NOT in the sea area, it's considered "land" for routing purposes
        return not self.sea_area.contains(point)

    def find_nearest_water(self, lat: float, lon: float, max_search_radius: float = 2.0) -> Tuple[float, float]:
        """Finds the closest point within the navigable sea area."""
        if not self.is_on_land(lat, lon):
            return lat, lon
        
        # Search in expanding circles (distance in degrees)
        for r in np.linspace(0.1, max_search_radius, 20):
            for angle in np.linspace(0, 2 * np.pi, 36): # Check every 10 degrees
                test_lon = lon + r * np.cos(angle)
                test_lat = lat + r * np.sin(angle)
                
                # Check if the new point is inside the defined sea area
                if not self.is_on_land(test_lat, test_lon):
                    return test_lat, test_lon
        
        print(f"⚠️  Could not find sea area within {max_search_radius} degrees of ({lat}, {lon})")
        return lat, lon # Fallback to original point

class WeatherDataProvider:
    """Provides weather data from multiple sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.synthetic_data_enabled = True
        self.land_mask = LandMaskService()
        
    def get_weather_data(self, lat: float, lon: float, timestamp: datetime = None) -> WeatherData:
        """Get weather data for a specific location and time"""
        if timestamp is None:
            timestamp = datetime.now()
            
        if self.api_key:
            try:
                return self._get_real_weather_data(lat, lon, timestamp)
            except Exception as e:
                print(f"Failed to get real weather data: {e}")
        
        return self._generate_synthetic_weather_data(lat, lon, timestamp)
    
    def _get_real_weather_data(self, lat: float, lon: float, timestamp: datetime) -> WeatherData:
        """Get real weather data from OpenWeatherMap API"""
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {'lat': lat, 'lon': lon, 'appid': self.api_key, 'units': 'metric'}
        response = requests.get(url, params=params)
        data = response.json()
        if response.status_code != 200:
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
        
        return WeatherData(
            lat=lat, lon=lon,
            wind_speed=data.get('wind', {}).get('speed', 0) * 1.94384,
            wind_direction=data.get('wind', {}).get('deg', 0),
            wave_height=self._estimate_wave_height(data.get('wind', {}).get('speed', 0)),
            wave_period=np.random.uniform(4, 12),
            wave_direction=data.get('wind', {}).get('deg', 0) + np.random.uniform(-30, 30),
            temperature=data.get('main', {}).get('temp', 20),
            visibility=data.get('visibility', 10000) / 1000,
            timestamp=timestamp
        )
    
    def _estimate_wave_height(self, wind_speed_ms: float) -> float:
        """Estimate wave height from wind speed using Beaufort scale approximation"""
        wind_speed_knots = wind_speed_ms * 1.94384
        if wind_speed_knots < 4: return np.random.uniform(0.1, 0.3)
        elif wind_speed_knots < 11: return np.random.uniform(0.3, 0.8)
        elif wind_speed_knots < 16: return np.random.uniform(0.8, 1.5)
        elif wind_speed_knots < 22: return np.random.uniform(1.5, 2.5)
        elif wind_speed_knots < 28: return np.random.uniform(2.5, 4.0)
        elif wind_speed_knots < 34: return np.random.uniform(4.0, 6.0)
        else: return np.random.uniform(6.0, 12.0)
    
    def _generate_synthetic_weather_data(self, lat: float, lon: float, timestamp: datetime) -> WeatherData:
        """Generate realistic synthetic weather data"""
        day_of_year = timestamp.timetuple().tm_yday
        season_factor = math.sin(2 * math.pi * day_of_year / 365)
        lat_factor = abs(lat) / 90.0
        monsoon_factor = 0
        if 5 <= lat <= 25 and 65 <= lon <= 95:
            if 150 <= day_of_year <= 270: monsoon_factor = 1.5
            elif day_of_year >= 275 or day_of_year <= 31: monsoon_factor = 1.2
        
        base_wind = 10 + lat_factor * 15 + monsoon_factor * 8
        base_wave = 1.5 + lat_factor * 2.5 + monsoon_factor * 1.5
        wind_speed = max(0, base_wind + np.random.normal(0, 5) + season_factor * 8)
        wave_height = max(0.1, base_wave + np.random.normal(0, 1) + season_factor * 1.5)
        
        return WeatherData(
            lat=lat, lon=lon, wind_speed=wind_speed,
            wind_direction=np.random.uniform(0, 360),
            wave_height=wave_height, wave_period=np.random.uniform(4, 14),
            wave_direction=np.random.uniform(0, 360),
            temperature=20 + season_factor * 15 - lat_factor * 20 + monsoon_factor * 5,
            visibility=max(1, min(20, 15 - monsoon_factor * 5 + np.random.normal(0, 3))),
            timestamp=timestamp
        )

class ShipRoutingModel:
    """Enhanced neural network model for ship routing optimization"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.land_mask = LandMaskService()
        
    def create_model(self) -> Model:
        """Create enhanced neural network architecture"""
        weather_input = Input(shape=(10,), name='weather_input')
        weather_dense = Dense(64, activation='relu')(weather_input)
        weather_dense = BatchNormalization()(weather_dense)
        weather_dense = Dropout(0.3)(weather_dense)
        weather_dense = Dense(32, activation='relu')(weather_dense)
        
        route_input = Input(shape=(10, 2), name='route_input')
        route_lstm = LSTM(64, return_sequences=True)(route_input)
        route_lstm = LSTM(32, return_sequences=True)(route_lstm)
        route_lstm = LSTM(16)(route_lstm)
        
        ship_input = Input(shape=(5,), name='ship_input')
        ship_dense = Dense(32, activation='relu')(ship_input)
        ship_dense = Dense(16, activation='relu')(ship_dense)
        
        env_input = Input(shape=(3,), name='env_input')
        env_dense = Dense(16, activation='relu')(env_input)
        
        combined = concatenate([weather_dense, route_lstm, ship_dense, env_dense])
        
        x = Dense(256, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        
        output = Dense(5, activation='linear', name='route_output')(x)
        model = Model(inputs=[weather_input, route_input, ship_input, env_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def prepare_training_data(self, weather_provider: WeatherDataProvider, num_samples: int = 20000) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate enhanced training data with 20,000+ samples"""
        print(f"Generating {num_samples} training samples with land avoidance...")
        weather_data, route_data, ship_data, env_data, targets = [], [], [], [], []
        
        major_routes = [
            {"start": (19.0760, 72.8777), "end": (25.2048, 55.2708)}, # Mumbai to Dubai
            {"start": (13.0827, 80.2707), "end": (1.3521, 103.8198)},# Chennai to Singapore
            {"start": (22.5726, 88.3639), "end": (22.3193, 114.1694)},# Kolkata to Hong Kong
            {"start": (9.9312, 76.2673), "end": (-20.1609, 57.5012)}, # Kochi to Port Louis
            {"start": (17.6868, 83.2185), "end": (16.8661, 96.1951)}, # Visakhapatnam to Yangon
            {"start": (19.0760, 72.8777), "end": (6.9271, 79.8612)},  # Mumbai to Colombo
        ]
        
        valid_samples, attempts, max_attempts = 0, 0, num_samples * 3
        
        while valid_samples < num_samples and attempts < max_attempts:
            attempts += 1
            if attempts % 1000 == 0: print(f"Generated {valid_samples} valid samples (attempts: {attempts})...")
            
            if np.random.random() < 0.7 and major_routes:
                route = random.choice(major_routes)
                start_lat, start_lon = route["start"]
                end_lat, end_lon = route["end"]
                start_lat += np.random.normal(0, 0.5); start_lon += np.random.normal(0, 0.5)
                end_lat += np.random.normal(0, 0.5); end_lon += np.random.normal(0, 0.5)
            else:
                start_lat, start_lon = np.random.uniform(-30, 30), np.random.uniform(50, 120)
                end_lat, end_lon = np.random.uniform(-30, 30), np.random.uniform(50, 120)
            
            start_lat, start_lon = weather_provider.land_mask.find_nearest_water(start_lat, start_lon)
            end_lat, end_lon = weather_provider.land_mask.find_nearest_water(end_lat, end_lon)
            
            route_context, current_lat, current_lon = [], start_lat, start_lon
            for j in range(10):
                progress = j / 10
                target_lat, target_lon = start_lat + (end_lat - start_lat) * progress, start_lon + (end_lon - start_lon) * progress
                dlat, dlon = (target_lat - current_lat) * 0.3 + np.random.uniform(-0.2, 0.2), (target_lon - current_lon) * 0.3 + np.random.uniform(-0.2, 0.2)
                current_lat += dlat; current_lon += dlon
                current_lat, current_lon = weather_provider.land_mask.find_nearest_water(current_lat, current_lon, max_search_radius=0.5)
                route_context.append([current_lat, current_lon])
            
            try: weather = weather_provider.get_weather_data(current_lat, current_lon)
            except: continue
            
            weather_features = [
                weather.wind_speed, weather.wind_direction / 360.0, weather.wave_height,
                weather.wave_period, weather.wave_direction / 360.0, weather.temperature,
                weather.visibility, math.sin(2 * math.pi * weather.timestamp.timetuple().tm_yday / 365),
                math.cos(2 * math.pi * weather.timestamp.timetuple().tm_hour / 24), abs(weather.lat) / 90.0
            ]
            
            ship_type = np.random.randint(0, 3)
            ship_features = [ship_type / 2.0, np.random.uniform(0.5, 2.0), np.random.uniform(8, 25) / 25.0, np.random.uniform(0, 1), np.random.uniform(0, 1)]
            env_features = [current_lat / 90.0, current_lon / 180.0, self._estimate_depth(current_lat, current_lon)]
            
            try:
                next_lat, next_lon, fuel_cost, time_cost, risk = self._calculate_optimal_waypoint_enhanced(
                    current_lat, current_lon, end_lat, end_lon, weather, ship_features, weather_provider.land_mask
                )
                next_lat, next_lon = weather_provider.land_mask.find_nearest_water(next_lat, next_lon)
                
                if self._is_valid_waypoint(current_lat, current_lon, next_lat, next_lon):
                    weather_data.append(weather_features)
                    route_data.append(route_context)
                    ship_data.append(ship_features)
                    env_data.append(env_features)
                    targets.append([next_lat / 90.0, next_lon / 180.0, fuel_cost, time_cost, risk])
                    valid_samples += 1
            except Exception: continue
        
        print(f"Training data generation complete! Generated {valid_samples} valid samples.")
        return [np.array(d) for d in [weather_data, route_data, ship_data, env_data]], np.array(targets)
    
    def _estimate_depth(self, lat: float, lon: float) -> float:
        """Estimate normalized ocean depth (0-1, 1 is deepest) based on distance to nearest land."""
        if self.land_mask.is_on_land(lat, lon): return 0.0
        if self.land_mask.sea_area is None: return 1.0
        
        # A proper implementation would find distance to the boundary of the sea_area polygon
        return 0.8 # Assume deep water for now if not on land

    def _is_valid_waypoint(self, current_lat, current_lon, next_lat, next_lon):
        if geodesic((current_lat, current_lon), (next_lat, next_lon)).kilometers > 500: return False
        if not (-90 <= next_lat <= 90 and -180 <= next_lon <= 180): return False
        return True
    
    def _calculate_optimal_waypoint_enhanced(self, lat, lon, end_lat, end_lon, weather, ship_features, land_mask):
        ship_type_idx, ship_size, ship_speed, cargo_weight, ship_age = ship_features
        dest_bearing = math.atan2(end_lon - lon, end_lat - lat)
        wind_resistance, wave_resistance = 1 + (weather.wind_speed / 40)**2, 1 + (weather.wave_height / 8)**2
        ship_efficiency, age_factor = [0.9, 0.85, 0.8][int(ship_type_idx * 2)], 1 + (ship_age * 0.2)
        preferred_direction = math.radians(weather.wind_direction) + math.pi
        weather_weight = min(weather.wind_speed / 30, 0.4)
        optimal_bearing = (1 - weather_weight) * dest_bearing + weather_weight * preferred_direction
        actual_bearing = optimal_bearing + np.random.normal(0, math.pi / 12)
        distance_to_dest = geodesic((lat, lon), (end_lat, end_lon)).kilometers
        base_step_km = min(100, distance_to_dest * 0.2)
        step_km = base_step_km / (wind_resistance * wave_resistance)
        step_lat = (step_km / 111.32) * math.cos(actual_bearing)
        step_lon = (step_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(actual_bearing)
        next_lat, next_lon = lat + step_lat, lon + step_lon
        
        attempts = 0
        while land_mask.is_on_land(next_lat, next_lon) and attempts < 8:
            alternative_bearing = optimal_bearing + (attempts - 4) * math.pi / 6
            step_lat, step_lon = (step_km / 111.32) * math.cos(alternative_bearing), (step_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(alternative_bearing)
            next_lat, next_lon = lat + step_lat, lon + step_lon
            attempts += 1
        
        if land_mask.is_on_land(next_lat, next_lon):
            next_lat, next_lon = land_mask.find_nearest_water(next_lat, next_lon)
        
        fuel_cost = (wind_resistance * wave_resistance * age_factor * (1 + cargo_weight * 0.3)) / ship_efficiency
        time_cost = fuel_cost / (ship_speed * ship_size)
        weather_risk, proximity_risk = min(1.0, (weather.wave_height / 6 + weather.wind_speed / 50) / 2), 0.5 if attempts > 0 else 0.0
        visibility_risk = max(0, (10 - weather.visibility) / 10)
        risk_factor = (weather_risk + proximity_risk + visibility_risk) / 3
        return next_lat, next_lon, fuel_cost, time_cost, risk_factor
    
    def train(self, weather_provider: WeatherDataProvider, num_samples: int = 20000):
        print(f"Training model with {num_samples} samples...")
        X, y = self.prepare_training_data(weather_provider, num_samples)
        self.model = self.create_model()
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X[0] = self.scaler.fit_transform(X[0])
        print("Starting enhanced model training...")
        self.model.fit(X, y, batch_size=64, epochs=100, validation_split=0.2, verbose=1, callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5)])
        self.is_trained = True
        print("Enhanced model training complete!")
    
    def predict_next_waypoint(self, weather_data: WeatherData, route_context: List[Tuple[float, float]], ship_features: List[float], current_pos: Tuple[float, float]) -> RoutePoint:
        if not self.is_trained or self.model is None: raise ValueError("Model must be trained before making predictions")
        
        weather_features = np.array([[
            weather_data.wind_speed, weather_data.wind_direction / 360.0, weather_data.wave_height, weather_data.wave_period, 
            weather_data.wave_direction / 360.0, weather_data.temperature, weather_data.visibility,
            math.sin(2*math.pi*weather_data.timestamp.timetuple().tm_yday/365), math.cos(2*math.pi*weather_data.timestamp.timetuple().tm_hour/24),
            abs(weather_data.lat) / 90.0 ]])
        weather_features = self.scaler.transform(weather_features)
        
        current_pos_list = list(current_pos) if current_pos else [weather_data.lat, weather_data.lon]
        while len(route_context) < 10: route_context.insert(0, current_pos_list)
        route_features = np.array([route_context[-10:]])
        
        ship_features_norm = np.array([[ship_features[0]/2.0, ship_features[1], ship_features[2]/25.0, ship_features[3], ship_features[4] if len(ship_features) > 4 else 0.5]])
        
        current_lat, current_lon = current_pos if current_pos else (weather_data.lat, weather_data.lon)
        env_features = np.array([[current_lat / 90.0, current_lon / 180.0, self._estimate_depth(current_lat, current_lon)]])
        
        prediction = self.model.predict([weather_features, route_features, ship_features_norm, env_features])
        next_lat, next_lon = prediction[0][0] * 90.0, prediction[0][1] * 180.0
        fuel_cost, time_cost, risk_factor = float(prediction[0][2]), float(prediction[0][3]), float(prediction[0][4])
        
        if self.land_mask.is_on_land(next_lat, next_lon):
            next_lat, next_lon = self.land_mask.find_nearest_water(next_lat, next_lon)
        
        return RoutePoint(lat=next_lat, lon=next_lon, weather=weather_data, fuel_cost=fuel_cost, time_cost=time_cost, risk_factor=risk_factor)
    
    def save_model(self, filepath: str):
        if self.model:
            self.model.save(f"{filepath}_model.h5")
            with open(f"{filepath}_scaler.pkl", 'wb') as f: pickle.dump(self.scaler, f)
            print(f"Enhanced model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model with a fix for deserialization errors."""
        try:
            self.model = tf.keras.models.load_model(f"{filepath}_model.h5", custom_objects={'mse': MeanSquaredError()})
            with open(f"{filepath}_scaler.pkl", 'rb') as f: self.scaler = pickle.load(f)
            self.is_trained = True
            print(f"Enhanced model loaded from {filepath}")
        except Exception as e:
            print(f"Failed to load model: {e}")

class OptimalRouteGenerator:
    """Generate optimal routes using the enhanced trained model"""
    
    def __init__(self, model: ShipRoutingModel, weather_provider: WeatherDataProvider):
        self.model, self.weather_provider, self.land_mask = model, weather_provider, LandMaskService()
    
    def generate_route(self, start_lat, start_lon, end_lat, end_lon, ship_type, departure_time=None):
        if departure_time is None: departure_time = datetime.now()
        start_lat, start_lon = self.land_mask.find_nearest_water(start_lat, start_lon)
        end_lat, end_lon = self.land_mask.find_nearest_water(end_lat, end_lon)
        
        ship_params = self._get_ship_parameters(ship_type)
        route, current_lat, current_lon, current_time = [], start_lat, start_lon, departure_time
        route_context = [(current_lat, current_lon)]
        max_iter, iteration, min_step = 200, 0, 50
        
        print(f"Generating route from ({start_lat:.3f}, {start_lon:.3f}) to ({end_lat:.3f}, {end_lon:.3f})")
        
        while iteration < max_iter:
            dist_to_dest = geodesic((current_lat, current_lon), (end_lat, end_lon)).kilometers
            if dist_to_dest < 50:
                weather = self.weather_provider.get_weather_data(end_lat, end_lon, current_time)
                route.append(RoutePoint(end_lat, end_lon, weather, 0.1, 0.1, 0.1))
                print(f"Route completed in {iteration} steps.")
                break
            
            weather = self.weather_provider.get_weather_data(current_lat, current_lon, current_time)
            try:
                next_point = self.model.predict_next_waypoint(weather, route_context, ship_params, (current_lat, current_lon))
                if geodesic((current_lat, current_lon), (next_point.lat, next_point.lon)).kilometers < min_step:
                    dir_to_dest = math.atan2(end_lon - current_lon, end_lat - current_lat)
                    step_km = max(min_step, dist_to_dest * 0.3)
                    next_lat = current_lat + (step_km / 111.32) * math.cos(dir_to_dest)
                    next_lon = current_lon + (step_km / (111.32 * math.cos(math.radians(current_lat)))) * math.sin(dir_to_dest)
                    next_point.lat, next_point.lon = self.land_mask.find_nearest_water(next_lat, next_lon)
                
                next_point = self._adjust_towards_destination(next_point, current_lat, current_lon, end_lat, end_lon, dist_to_dest)
                route.append(next_point)
                route_context.append((next_point.lat, next_point.lon))
                current_lat, current_lon = next_point.lat, next_point.lon
                current_time += timedelta(hours=next_point.time_cost)
                iteration += 1
                if iteration % 10 == 0: print(f"Step {iteration}: Pos({current_lat:.3f}, {current_lon:.3f}), {dist_to_dest:.1f} km to dest")
            except Exception as e:
                print(f"Error at iteration {iteration}: {e}")
                break
        
        if iteration >= max_iter: print("Warning: Max iterations reached.")
        optimized_route = self._optimize_waypoints(route)
        print(f"Route optimized: {len(route)} -> {len(optimized_route)} waypoints")
        return optimized_route
    
    def _optimize_waypoints(self, route):
        if len(route) <= 2: return route
        optimized = [route[0]]
        i = 0
        while i < len(route) - 1:
            current = optimized[-1]
            max_direct_index = i + 1
            for j in range(i + 2, min(i + 10, len(route))):
                if self._is_direct_path_clear(current.lat, current.lon, route[j].lat, route[j].lon): max_direct_index = j
                else: break
            if max_direct_index < len(route):
                optimized.append(route[max_direct_index])
                i = max_direct_index
            else: break
        if not optimized or optimized[-1] != route[-1]: optimized.append(route[-1])
        return optimized
    
    def _is_direct_path_clear(self, lat1, lon1, lat2, lon2):
        num_samples = max(3, int(geodesic((lat1, lon1), (lat2, lon2)).kilometers / 50))
        for i in range(1, num_samples):
            t = i / num_samples
            sample_lat, sample_lon = lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t
            if self.land_mask.is_on_land(sample_lat, sample_lon): return False
        return True
    
    def _calculate_total_distance(self, route):
        if len(route) < 2: return 0.0
        return sum(geodesic((route[i].lat, route[i].lon), (route[i+1].lat, route[i+1].lon)).kilometers for i in range(len(route)-1))
    
    def _get_ship_parameters(self, ship_type):
        params = {"passenger ship": [0,1.0,20,0.3,0.3], "cargo ship": [1,1.5,15,0.8,0.5], "tanker": [2,2.0,12,1.0,0.6]}
        return params.get(ship_type.lower(), [1,1.0,15,0.5,0.5])
    
    def _adjust_towards_destination(self, point, cur_lat, cur_lon, dest_lat, dest_lon, dist):
        dest_bearing = math.atan2(dest_lon - cur_lon, dest_lat - cur_lat)
        pred_bearing = math.atan2(point.lon - cur_lon, point.lat - cur_lat)
        
        if dist < 500: weight = 0.8
        elif dist < 1000: weight = 0.6
        else: weight = 0.4
        
        blend_bearing = weight * dest_bearing + (1 - weight) * pred_bearing
        step_dist = max(50, min(geodesic((cur_lat, cur_lon), (point.lat, point.lon)).kilometers, dist * 0.4))
        
        new_lat = cur_lat + (step_dist / 111.32) * math.cos(blend_bearing)
        new_lon = cur_lon + (step_dist / (111.32 * math.cos(math.radians(cur_lat)))) * math.sin(blend_bearing)
        point.lat, point.lon = self.land_mask.find_nearest_water(new_lat, new_lon)
        return point

if __name__ == "__main__":
    land_mask_path = "land_polygons.pkl"
    if not os.path.exists(land_mask_path):
        print(f"FATAL ERROR: The required land mask file '{land_mask_path}' was not found.")
    else:
        print("Initializing Enhanced Ship Routing ML System...")
        weather_provider = WeatherDataProvider()
        model = ShipRoutingModel()
        model_path = "enhanced_ship_routing_model"
        if os.path.exists(f"{model_path}_model.h5"):
            print("Found pre-trained model. Loading...")
            model.load_model(model_path)
        else:
            print("No pre-trained model found. Training a new one (this will take 5-10 minutes)...")
            model.train(weather_provider, num_samples=20000)
            model.save_model(model_path)
        
        route_generator = OptimalRouteGenerator(model, weather_provider)
        print("\nGenerating test route from Mumbai to Dubai...")
        route = route_generator.generate_route(19.0760, 72.8777, 25.2048, 55.2708, "cargo ship")
        
        print(f"\nGenerated optimized route with {len(route)} waypoints:")
        for i, p in enumerate(route[:10]): print(f"Waypoint {i+1}: ({p.lat:.4f}, {p.lon:.4f}), Risk: {p.risk_factor:.3f}")
        
        total_dist = route_generator._calculate_total_distance(route)
        print(f"\nTotal route distance: {total_dist:.1f} km")
        if len(route) > 1: print(f"Average waypoint separation: {total_dist/max(1, len(route)-1):.1f} km")
        
        print("\nEnhanced Ship Routing ML System ready!")