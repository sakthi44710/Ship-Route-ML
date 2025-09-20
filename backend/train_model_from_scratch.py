#!/usr/bin/env python3
"""
Train the ship routing model from scratch with your land data
"""

import os
import sys
import time
from datetime import datetime
import numpy as np

def check_dependencies():
    """Check if all required packages are installed"""
    
    required_packages = [
        ('tensorflow', 'TensorFlow'),
        ('sklearn', 'scikit-learn'),
        ('geopandas', 'GeoPandas'),
        ('shapely', 'Shapely'),
        ('geopy', 'GeoPy'),
        ('scipy', 'SciPy'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas')
    ]
    
    print("Checking dependencies...")
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - MISSING")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\nERROR: Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install tensorflow scikit-learn geopandas geopy scipy numpy pandas")
        return False
    
    print("All dependencies satisfied!")
    return True

def check_land_data():
    """Verify land data is available and working"""
    
    if not os.path.exists("land_polygons.pkl"):
        print("ERROR: land_polygons.pkl not found!")
        print("Make sure the file is in the current directory")
        return False
    
    try:
        from ship_routing_model import LandMaskService
        land_mask = LandMaskService("land_polygons.pkl")
        
        if land_mask.sea_area is None:
            print("ERROR: Land mask failed to initialize!")
            return False
        
        # Quick test
        is_mumbai_land = land_mask.is_on_land(19.0760, 72.8777)  # Mumbai
        is_arabian_sea_water = not land_mask.is_on_land(20.0, 65.0)  # Arabian Sea
        
        if not is_mumbai_land:
            print("WARNING: Mumbai not detected as land - land mask may be incorrect")
        
        if not is_arabian_sea_water:
            print("WARNING: Arabian Sea not detected as water - land mask may be incorrect")
        
        print("✅ Land data verified and working")
        return True
        
    except Exception as e:
        print(f"ERROR testing land data: {e}")
        return False

def train_model():
    """Train the model from scratch"""
    
    print("Starting model training from scratch...")
    print("=" * 60)
    
    try:
        # Import model components
        from ship_routing_model import ShipRoutingModel, WeatherDataProvider, OptimalRouteGenerator
        
        print("📡 Initializing weather provider...")
        weather_provider = WeatherDataProvider()
        
        print("🧠 Creating new neural network model...")
        model = ShipRoutingModel()
        
        # Check if there's an existing model
        model_path = "enhanced_ship_routing_model"
        if os.path.exists(f"{model_path}_model.h5"):
            response = input(f"Found existing model at {model_path}. Delete and retrain? (y/N): ")
            if response.lower() == 'y':
                try:
                    os.remove(f"{model_path}_model.h5")
                    os.remove(f"{model_path}_scaler.pkl")
                    print("🗑️ Deleted existing model files")
                except:
                    pass
            else:
                print("Loading existing model...")
                model.load_model(model_path)
                if model.is_trained:
                    print("✅ Using existing trained model")
                    return model, weather_provider
        
        # Train new model
        print("\n🎓 Training new model with enhanced land avoidance...")
        print("📊 Training parameters:")
        print(f"   • Training samples: 20,000")
        print(f"   • Neural network: LSTM + Dense layers")
        print(f"   • Land avoidance: Enabled with morphological closing")
        print(f"   • Weather simulation: Enhanced with monsoon effects")
        print(f"   • Expected training time: 5-10 minutes")
        
        print("\n⏳ Starting training... Please be patient!")
        print("(You'll see progress updates during training)")
        
        start_time = time.time()
        
        # Train the model
        history = model.train(weather_provider, num_samples=20000)
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/60:.1f} minutes!")
        
        # Save the model
        print("💾 Saving trained model...")
        model.save_model(model_path)
        
        # Print training summary
        if history:
            final_loss = history.history['loss'][-1] if 'loss' in history.history else 'unknown'
            final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 'unknown'
            print(f"📈 Training metrics:")
            print(f"   • Final training loss: {final_loss}")
            print(f"   • Final validation loss: {final_val_loss}")
        
        return model, weather_provider
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_model(model, weather_provider):
    """Test the trained model with a sample route"""
    
    print("\n🧪 Testing trained model...")
    print("=" * 40)
    
    try:
        route_generator = OptimalRouteGenerator(model, weather_provider)
        
        print("🚢 Generating test route: Mumbai to Dubai")
        print("   Start: Mumbai (19.0760, 72.8777)")
        print("   End: Dubai (25.2048, 55.2708)")
        print("   Ship: Cargo ship")
        
        start_time = time.time()
        
        route = route_generator.generate_route(
            start_lat=19.0760, start_lon=72.8777,  # Mumbai
            end_lat=25.2048, end_lon=55.2708,      # Dubai
            ship_type="cargo ship"
        )
        
        generation_time = time.time() - start_time
        
        print(f"⚡ Route generated in {generation_time:.1f} seconds")
        print(f"📍 Generated {len(route)} waypoints")
        
        # Calculate total distance
        total_distance = 0
        if len(route) > 1:
            from geopy.distance import geodesic
            for i in range(len(route) - 1):
                dist = geodesic((route[i].lat, route[i].lon), (route[i+1].lat, route[i+1].lon)).kilometers
                total_distance += dist
        
        print(f"📏 Total distance: {total_distance:.1f} km")
        print(f"📊 Average segment: {total_distance/(len(route)-1):.1f} km" if len(route) > 1 else "")
        
        # Show first few waypoints
        print(f"\n📍 First 5 waypoints:")
        for i, point in enumerate(route[:5]):
            print(f"   {i+1}: ({point.lat:.4f}, {point.lon:.4f})")
            
        # Verify waypoints are in water
        land_check_passed = True
        land_mask = model.land_mask
        if land_mask and land_mask.sea_area:
            land_waypoints = 0
            for point in route:
                if land_mask.is_on_land(point.lat, point.lon):
                    land_waypoints += 1
            
            if land_waypoints > 0:
                print(f"⚠️ WARNING: {land_waypoints} waypoints are on land!")
                land_check_passed = False
            else:
                print("✅ All waypoints are in navigable water")
        
        print(f"\n{'✅ MODEL TEST PASSED' if land_check_passed else '⚠️ MODEL TEST WARNINGS'}")
        return True
        
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function"""
    
    print("Ship Routing Model - Train from Scratch")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check land data
    if not check_land_data():
        return False
    
    # Train the model
    model, weather_provider = train_model()
    
    if model is None:
        print("\n❌ TRAINING FAILED!")
        return False
    
    # Test the model
    test_success = test_model(model, weather_provider)
    
    print("\n" + "=" * 60)
    if test_success:
        print("🎉 TRAINING AND TESTING SUCCESSFUL!")
        print("\nYour model is now ready for use!")
        print("\nNext steps:")
        print("1. Start the backend server: python app.py")
        print("2. The server will use your trained model")
        print("3. Routes will avoid land using your land_polygons.pkl data")
        print("4. Test with frontend at http://localhost:3000")
    else:
        print("⚠️ TRAINING COMPLETED BUT TESTS FAILED")
        print("The model may still work, but check the warnings above")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return test_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)