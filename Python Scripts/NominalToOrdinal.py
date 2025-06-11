import pandas as pd
import numpy as np

binding_strength_map = {
    'E' : 4,  'E1': 4,
    'G' : 3,  'G1': 3,
    'M' : 2,  'M1': 2,
    'W' : 1,  'W1': 1,
    'Z' : 0,  'N' :0
}

def convert_to_ordinal(df):
    return df.replace(binding_strength_map)

# Load your original dataset
df = pd.read_csv("data/final_data.csv")

# Apply mapping to all binding-strength columns
ordinal_df = convert_to_ordinal(df)

# Save as CSV (for Weka)
ordinal_df.to_csv("data/ordinal_encoded_data_grouped.csv", index=False)