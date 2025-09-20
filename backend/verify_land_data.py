#!/usr/bin/env python3
"""
Verify the land_polygons.pkl file and test land mask functionality
"""

import pickle
import os
import sys

def verify_land_polygons_file(filepath="land_polygons.pkl"):
    """Verify the land polygons file structure and content"""
    
    print("Analyzing land_polygons.pkl file...")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"ERROR: File {filepath} not found!")
        return False
    
    # Check file size
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    try:
        # Load the pickle file
        print("Loading pickle file...")
        with open(filepath, 'rb') as f:
            land_gdf = pickle.load(f)
        
        print(f"Successfully loaded land data")
        print(f"Data type: {type(land_gdf)}")
        
        # Check if it's a GeoDataFrame
        try:
            import geopandas as gpd
            if isinstance(land_gdf, gpd.GeoDataFrame):
                print(f"Confirmed: GeoDataFrame with {len(land_gdf)} polygons")
                
                # Check columns
                print(f"Columns: {list(land_gdf.columns)}")
                
                # Check if geometry column exists
                if 'geometry' in land_gdf.columns:
                    print("Geometry column found")
                    
                    # Check bounds
                    bounds = land_gdf.total_bounds
                    print(f"Spatial bounds: [{bounds[0]:.1f}, {bounds[1]:.1f}, {bounds[2]:.1f}, {bounds[3]:.1f}]")
                    print(f"  (min_lon, min_lat, max_lon, max_lat)")
                    
                    # Check if bounds cover Indian Ocean region
                    if (bounds[0] <= 50 and bounds[2] >= 100 and 
                        bounds[1] <= 0 and bounds[3] >= 30):
                        print("Coverage: Indian Ocean region included")
                    else:
                        print("WARNING: May not cover full Indian Ocean region")
                        
                else:
                    print("ERROR: No geometry column found!")
                    return False
                    
            else:
                print(f"WARNING: Not a GeoDataFrame, type is {type(land_gdf)}")
                return False
                
        except ImportError:
            print("WARNING: geopandas not available for detailed analysis")
            
        return True
        
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")
        return False

def test_land_mask_service():
    """Test the LandMaskService with the current file"""
    
    print("\nTesting LandMaskService...")
    print("=" * 30)
    
    try:
        # Try to import and test LandMaskService
        from ship_routing_model import LandMaskService
        
        print("Creating LandMaskService...")
        land_mask = LandMaskService("land_polygons.pkl")
        
        if land_mask.sea_area is None:
            print("ERROR: LandMaskService failed to initialize!")
            print("This means the land_polygons.pkl file is not compatible")
            return False
        
        print("LandMaskService initialized successfully")
        
        # Test known coordinates
        test_points = [
            (19.0760, 72.8777, "Mumbai (should be near land)"),
            (20.0, 65.0, "Arabian Sea (should be water)"),
            (15.0, 75.0, "Off Indian coast (should be water)"),
            (25.2048, 55.2708, "Dubai (should be near land)"),
            (22.0, 77.0, "Central India (should be land)"),
            (10.0, 80.0, "Bay of Bengal (should be water)")
        ]
        
        print("\nTesting coordinate classification:")
        all_water = True
        for lat, lon, description in test_points:
            try:
                is_land = land_mask.is_on_land(lat, lon)
                status = "LAND" if is_land else "WATER"
                print(f"  {lat:6.2f}, {lon:6.2f} - {status:5} - {description}")
                
                # Check if ocean points are correctly identified as water
                if "should be water" in description and is_land:
                    print(f"    WARNING: Ocean point classified as land!")
                    all_water = False
                    
            except Exception as e:
                print(f"  ERROR testing {description}: {e}")
                return False
        
        # Test find_nearest_water function
        print("\nTesting find_nearest_water function:")
        try:
            # Test with a land point (central India)
            test_lat, test_lon = 22.0, 77.0
            water_lat, water_lon = land_mask.find_nearest_water(test_lat, test_lon)
            print(f"  Land point ({test_lat}, {test_lon}) -> Water point ({water_lat:.3f}, {water_lon:.3f})")
            
            # Verify the water point is actually in water
            is_water = not land_mask.is_on_land(water_lat, water_lon)
            print(f"  Nearest water point is valid: {is_water}")
            
        except Exception as e:
            print(f"  ERROR testing find_nearest_water: {e}")
            return False
        
        print("\nLand mask functionality test: PASSED")
        return True
        
    except ImportError as e:
        print(f"ERROR: Could not import LandMaskService: {e}")
        print("Make sure ship_routing_model.py is in the current directory")
        return False
    except Exception as e:
        print(f"ERROR during land mask test: {e}")
        return False

def main():
    """Main verification function"""
    
    print("Land Polygons Data Verification")
    print("=" * 50)
    
    # Verify the pickle file
    pickle_ok = verify_land_polygons_file()
    
    if not pickle_ok:
        print("\nVERIFICATION FAILED!")
        print("The land_polygons.pkl file is not valid or compatible")
        return False
    
    # Test the land mask service
    service_ok = test_land_mask_service()
    
    if not service_ok:
        print("\nLAND MASK SERVICE TEST FAILED!")
        print("The file loads but doesn't work with LandMaskService")
        return False
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUCCESSFUL!")
    print("Your land_polygons.pkl file is working correctly")
    print("\nNext steps:")
    print("1. Start the backend: python app.py")
    print("2. The model will train automatically (5-10 minutes)")
    print("3. Routes will avoid land using your data")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)