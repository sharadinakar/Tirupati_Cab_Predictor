# --- 1. LIBRARIES (All engines at the top) ---
import pandas as pd
from sklearn.model_selection import train_test_split

# --- 2. DATA LOADING ---
print("Loading 350MB dataset...")
df = pd.read_csv("data/uber_data.csv")

# --- 3. DATA CLEANING ---
print("Cleaning missing prices...")
df_clean = df.dropna(subset=['price'])

# --- 4. FEATURE SELECTION ---
core_columns = ['price', 'distance', 'cab_type', 'name', 'surge_multiplier']
df_model = df_clean[core_columns]

# --- 5. DATA ENCODING (Text to Math) ---
print("Encoding text to numbers...")
df_encoded = pd.get_dummies(df_model, columns=['cab_type', 'name'])

# --- 6. TARGET VS INPUTS (X and y) ---
y = df_encoded['price'] # Target
X = df_encoded.drop('price', axis=1) # Inputs

# --- 7. TRAIN/TEST SPLIT (80-20 Rule) ---
print("Splitting data into Training and Testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Final Pipeline Status ---")
print(f"Training Data (For the Engine to Learn): {X_train.shape[0]} rows")
print(f"Testing Data (For the Final Exam): {X_test.shape[0]} rows")