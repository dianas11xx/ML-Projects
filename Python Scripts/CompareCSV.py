import pandas as pd

# Load the two CSV files
csv1 = pd.read_csv('data/ordinal_encoded_data_grouped.csv', index_col=0)
csv2 = pd.read_csv('../ML Data/final_data_ordinal.csv', index_col=0)

# --- Search-based Column Check ---
columns1 = list(csv1.columns)
columns2 = list(csv2.columns)

missing_columns_in_csv2 = [col for col in columns1 if col not in columns2]
extra_columns_in_csv2 = [col for col in columns2 if col not in columns1]

print("=== Column Comparison ===")
print(f"Missing columns in file2 (from file1): {missing_columns_in_csv2}")
print(f"Extra columns in file2 (not in file1): {extra_columns_in_csv2}")

# --- Search-based Row Check ---
rows1 = list(csv1.index)
rows2 = list(csv2.index)

missing_rows_in_csv2 = [row for row in rows1 if row not in rows2]
extra_rows_in_csv2 = [row for row in rows2 if row not in rows1]

print("\n=== Row Comparison ===")
print(f"Missing rows in file2 (from file1): {missing_rows_in_csv2}")
print(f"Extra rows in file2 (not in file1): {extra_rows_in_csv2}")

print(f"Number of Missing columns in file2 (from file1): {len(missing_columns_in_csv2)}")
print(f"Number of extra columns in file 2: {len(extra_columns_in_csv2)}")
print(f"Number of Missing rows in file2 (from file1): {len(missing_rows_in_csv2)}")
print(f"Number of extra rows in file 2: {len(extra_rows_in_csv2)}")

