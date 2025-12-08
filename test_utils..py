from src.utils import data, metrics, features, viz
import numpy as np
print("All utils modules imported successfully!")

try:
    gold_data = data.load_data("gold", "processed")
    print(f"Loaded gold  shape {gold_data.shape}")
except Exception as e:
    print(f"Error loading  {e}")

print("Utils folder created successfully!")
