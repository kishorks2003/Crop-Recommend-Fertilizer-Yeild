import pickle

# Function to safely load and print the type of each pickle file
def load_pickle(filename):
    try:
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
            print(f"Loaded {filename}: {type(obj)}\n")
            return obj
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Load and print details of all .pkl files
crop_model = load_pickle('models/crop_model.pkl')
fertilizer_model = load_pickle('models/fertilizer_model.pkl')
scaler = load_pickle('models/scaler.pkl')
yield_model = load_pickle('models/yield_model.pkl')

# Print object details if successfully loaded
print("\n--- Model Details ---")
print(f"Crop Model: {crop_model}")
print(f"Fertilizer Model: {fertilizer_model}")
print(f"Scaler: {scaler}")
print(f"Yield Model: {yield_model}")
