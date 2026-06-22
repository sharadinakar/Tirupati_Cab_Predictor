import pandas as pd
import numpy as np

# Set random seed: every time you run this code, you'll get the same random numbers. This is useful for reproducibility.
np.random.seed(42)

# Total records (the size of our dummy dataset)
num_records = 500

# 1. Distance: 1 KM to 50 KM
distance = np.random.uniform(1, 50, num_records).round(2)

# 2. Traffic: 1 (free road) to 5 (full traffic jam)
traffic = np.random.randint(1, 6, num_records)

# 3. Weather: 0 (normal weather) or 1 (rainy)
weather = np.random.randint(0, 2, num_records)

# 4. The Logic (the actual logic our AI will use)
base_fare = 50
price = (
    base_fare + 
    (15 * distance) + 
    (10 * traffic) + 
    (30 * weather) + 
    np.random.normal(0, 10, num_records) # real-world data won't be perfect, so we add some noise (Noise)
)

# price is rounded to two decimal places
price = price.round(2)

# Pandas DataFrame creation
data = {
    'Distance_KM': distance,
    'Traffic_Level': traffic,
    'Weather_Rainy': weather,
    'Cab_Price_INR': price
}

df = pd.DataFrame(data)

# let's save this as a CSV file for future use or analysis.
df.to_csv('tirupati_cab_data.csv', index=False)

print("Boom! 500 records of Tirupati Cab Data generated successfully.")
print("-" * 50)
print(df.head()) # Print the first 5 rows of the DataFrame