import os
import pandas as pd
from pathlib import Path

#function to read files in a loop
def read_files_to_dataframe(folder_path, file_extension='.csv', **kwargs):

    # Create a Path object
    folder = Path(folder_path)

    # Initialize an empty list to store individual DataFrames
    dfs = []

    # Select the appropriate pandas read function based on file extension
    if file_extension.lower() == '.csv':
        read_func = pd.read_csv
    elif file_extension.lower() in ['.xlsx', '.xls']:
        read_func = pd.read_excel
    elif file_extension.lower() == '.json':
        read_func = pd.read_json
    elif file_extension.lower() == '.parquet':
        read_func = pd.read_parquet
    elif file_extension.lower() == '.pickle' or file_extension.lower() == '.pkl':
        read_func = pd.read_pickle
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    # Loop through all files in the folder
    for file_path in folder.glob(f'*{file_extension}'):
        try:
            # Read the file into a DataFrame
            df = read_func(file_path, **kwargs)

            # Add a column indicating the source file if needed
            df['source_file'] = file_path.name

            # Append to our list
            dfs.append(df)

            print(f"Successfully read: {file_path.name}")
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    # If no DataFrames were added to the list, return an empty DataFrame
    if len(dfs) == 0:
        print(f"No{file_extension} files found in {folder_path}")
        return pd.DataFrame()

    # Combine all DataFrames in the list
    combined_df = pd.concat(dfs, ignore_index=True)

    print(
        f"Combined {len(dfs)} files into a DataFrame with {combined_df.shape[0]} rows and {combined_df.shape[1]} columns")

    return combined_df

#read the QB files in
qb = read_files_to_dataframe(
    r"C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\Weekly QB Results FantasyData", ".xlsx")
print(qb.head())