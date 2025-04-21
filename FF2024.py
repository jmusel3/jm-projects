import os
import pandas as pd
from pathlib import Path
import numpy as np

#function to read files in a loop--------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------------------------------------

#QB data cleaning-----------------------------------------------------------------------------------
qb = read_files_to_dataframe(
    r"C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\Weekly QB Results FantasyData", ".xlsx")

# Extract first four characters, convert to integer, and create a new column named 'Year'
qb['Year'] = qb['source_file'].str[:4].astype(int)

# Drop un-needed columns, check output
qb = qb.drop(columns=['source_file','RK'])
print(qb.head())
print(qb.dtypes)

qb['Rank'] = qb.groupby(['Year', 'WK'])['FPTS'].rank(method='dense', ascending=False)

# Step 1: Create parv_step with selected columns from qb
parv_step = qb[['Rank', 'Year', 'FPTS']].copy()

# Step 2: Filter for rows where Rank equals 12
parv_step = parv_step[parv_step['Rank'] == 12]

# Step 3: Group by Year and calculate average FPTS
avg_by_year = parv_step.groupby('Year')['FPTS'].mean().reset_index()
avg_by_year.rename(columns={'FPTS': 'Replacement Level Points'}, inplace=True)

# check the result
print(avg_by_year)

#join in replacement level points to main QB dataframe
qb = qb.merge(avg_by_year, on='Year', how='inner')
qb['PARV']= qb['FPTS'] - qb['Replacement Level Points']
qb['PARV'] = np.where(qb['PARV'] < 0, 0, qb['PARV'])

print(qb.head())