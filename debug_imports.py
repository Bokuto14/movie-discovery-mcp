import sys
import os

print("Current working directory:", os.getcwd())
print("Python path:")
for p in sys.path:
    print(f"  {p}")

# Add project root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("\nAfter adding project root:")
for p in sys.path:
    print(f"  {p}")

# Test import
try:
    from src.models.database import movie_db
    print("\n✅ Import successful!")
except Exception as e:
    print(f"\n❌ Import failed: {e}")

# Check if files exist
print(f"\nsrc/models/database.py exists: {os.path.exists('src/models/database.py')}")
print(f"src/server.py exists: {os.path.exists('src/server.py')}")