import pandas as pd

# Load the Parquet file
df = pd.read_parquet(
    "/scratch/izar/nowak/gesture/data/asl-signs/train_landmark_files/16069/23173099.parquet"
)

df = pd.read_parquet(
    "/scratch/izar/nowak/gesture/data/wlaslvideos_processed/crocodile/65425_landmarks.parquet"
)

# See the first few rows
print(df)