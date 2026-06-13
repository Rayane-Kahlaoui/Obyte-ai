import pandas as pd, json, sys, pathlib

file_path = pathlib.Path(r'C:/Users/rayan/Desktop/Orbyte/data/train-00000-of-00001.parquet')
if not file_path.exists():
    print('Parquet file not found')
    sys.exit(1)

# Load parquet
try:
    df = pd.read_parquet(file_path)
except Exception as e:
    print('Error loading parquet:', e)
    sys.exit(1)

# Print column names
print('COLUMNS:', list(df.columns))

# Show a sample row
sample = df.iloc[0]
print('SAMPLE ROW:', sample.to_dict())

# Show first 5 rows as JSON
print('FIRST5_JSON:', df.head().to_json(orient='records'))
