#!/usr/bin/env python3
"""
Land Data Setup Script for Ship Routing System
Downloads Natural Earth data and creates the land_polygons.pkl file
"""

import os
import requests
import zipfile
import geopandas as gpd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def download_natural_earth_data(data_dir="./data", resolution="10m"):
    """Download and extract Natural Earth land data"""
    
    # Create data directory
    Path(data_dir).mkdir(exist_ok=True)
    
    filename = f'ne_{resolution}_land'
    shapefile_path = os.path.join(data_dir, f'{filename}.shp')
    
    # Check if already downloaded
    if os.path.exists(shapefile_path):
        print(f"✅ Natural Earth data already exists: {shapefile_path}")
        return shapefile_path
    
    # Download URL
    url = f'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/{resolution}/physical/{filename}.zip'
    zip_path = os.path.join(data_dir, f'{filename}.zip')
    
    try:
        print(f"⬇️  Downloading Natural Earth land data ({resolution})...")
        print(f"URL: {url}")
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Save zip file
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        print(f"📦 Extracting data to {data_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        
        if os.path.exists(shapefile_path):
            print(f"✅ Successfully downloaded: {shapefile_path}")
            return shapefile_path
        else:
            print("❌ Shapefile not found after extraction")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading Natural Earth data: {e}")
        return None

def create_land_polygons_file():
    """Create the land_polygons.pkl file required by the ship routing system"""
    
    print("🌍 Setting up land mask data for ship routing...")
    
    # Download Natural Earth data
    shapefile_path = download_natural_earth_data()
    
    if not shapefile_path:
        print("❌ Could not download Natural Earth data")
        return False
    
    try:
        print("📂 Loading land polygons...")
        land_gdf = gpd.read_file(shapefile_path)
        
        print(f"📊 Loaded {len(land_gdf)} land polygons")
        
        # Filter to focus on Indian Ocean region for better performance
        # Bounding box: longitude 30-130, latitude -40 to 40
        bbox_filter = (
            (land_gdf.bounds.minx < 130) & 
            (land_gdf.bounds.maxx > 30) & 
            (land_gdf.bounds.miny < 40) & 
            (land_gdf.bounds.maxy > -40)
        )
        
        land_gdf_filtered = land_gdf[bbox_filter].copy()
        print(f"🔍 Filtered to {len(land_gdf_filtered)} polygons in Indian Ocean region")
        
        # Simplify geometries slightly for better performance
        print("⚡ Simplifying geometries for performance...")
        land_gdf_filtered['geometry'] = land_gdf_filtered.geometry.simplify(0.01)
        
        # Save to pickle file
        pickle_path = "land_polygons.pkl"
        print(f"💾 Saving land polygons to {pickle_path}...")
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(land_gdf_filtered, f)
        
        print(f"✅ Successfully created {pickle_path}")
        print(f"📏 File size: {os.path.getsize(pickle_path) / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating land polygons file: {e}")
        return False

def verify_land_mask():
    """Test the land mask functionality"""
    
    try:
        print("\n🧪 Testing land mask functionality...")
        
        # Import the LandMaskService
        import sys
        sys.path.append('.')  # Add current directory to path
        
        from ship_routing_model import LandMaskService
        
        # Create land mask service
        land_mask = LandMaskService("land_polygons.pkl")
        
        if land_mask.sea_area is None:
            print("❌ Land mask service failed to initialize")
            return False
        
        # Test some known points
        test_points = [
            (19.0760, 72.8777, "Mumbai (should be near land)"),
            (20.0, 65.0, "Arabian Sea (should be in water)"),
            (15.0, 75.0, "West of India (should be in water)"),
            (20.0, 77.0, "Central India (should be on land)"),
            (10.0, 80.0, "Bay of Bengal (should be in water)")
        ]
        
        print("\n📍 Testing coordinates:")
        for lat, lon, description in test_points:
            is_land = land_mask.is_on_land(lat, lon)
            status = "🏔️  LAND" if is_land else "🌊 WATER"
            print(f"  {description}: {status}")
        
        print("\n✅ Land mask verification complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

def main():
    """Main setup function"""
    
    print("🚢 Ship Routing System - Land Data Setup")
    print("=" * 50)
    
    # Check if land_polygons.pkl already exists
    if os.path.exists("land_polygons.pkl"):
        print("📁 land_polygons.pkl already exists")
        
        response = input("Do you want to recreate it? (y/N): ").lower().strip()
        if response != 'y':
            print("✅ Using existing land_polygons.pkl")
            verify_land_mask()
            return
    
    # Create the land polygons file
    success = create_land_polygons_file()
    
    if success:
        # Verify it works
        verify_land_mask()
        
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Run your backend: python app.py")
        print("2. The system will now have accurate land avoidance")
        print("3. Routes will stay in navigable waters")
    else:
        print("\n❌ Setup failed!")
        print("The system will use fallback land detection")

if __name__ == "__main__":
    main()