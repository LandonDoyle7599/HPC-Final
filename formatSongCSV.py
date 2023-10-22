import pandas as pd

# Read the CSV file
df = pd.read_csv('tracks_features.csv')

# Select only the desired columns
columns_to_keep = ['energy', 'speechiness', 'instrumentalness']
df = df[columns_to_keep]

# Write the result to a new CSV file
df.to_csv('song_data.csv', index=False)