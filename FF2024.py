import os
import pandas as pd
from pathlib import Path
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


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

#start process of calculating PARV (Points Above Replacement Value)
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
print(len(qb)," before Taysom filter")

#Taysom is an outlier, filter out
qb = qb[qb['NAME'] != 'Taysom Hill']
print(len(qb))

#clean up name so it is consistent YOY
qb['NAME'] = np.where(qb['NAME'] =='Mitchell Trubisky','Mitch Trubisky',qb['NAME'])

qb = qb.groupby(['NAME','Year'])['PARV'].sum().reset_index()
#print(qb.head(), " after groupby")
#print(len(qb), " after groupby")

qb_games = read_files_to_dataframe(r'C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\QB results- FantasyPros')
print(qb_games.columns)
qb_games['Year'] = qb_games['source_file'].str[:4].astype(int)
qb_games = qb_games[['Player', 'G', 'Year']]
qb_games = qb_games.dropna(subset=['Player'])

qb_games['Player'] = qb_games['Player'].apply(
    lambda x: "Mitch Trubisky" if x == "Mitchell Trubisky"
    else "Tommy Devito" if x == "Tommy DeVito"
    else "Joe Webb" if x == "Joe WebbI"
    else "Robert Griffin" if x == "Robert GriffinI"
    else x
)
#print("Distinct players in the dataset:")
#print(sorted(qb_games['Player'].unique()))

qb_games.rename(columns={'Player': 'NAME'}, inplace=True)

qb = pd.merge(qb, qb_games, on=['NAME','Year'],how='inner')
print(qb.head())

qb = qb[qb['G'] > 0]
qb['PARVPG'] = qb['PARV'] / qb['G']
qb = qb.sort_values(by=['NAME', 'Year'], ascending=[True, True])
qb_games.rename(columns={'G': 'Current Season Games Played','PARV':'Current_Season_PARV'}, inplace=True)

#PARVPG LOOP-----------------------------------------------------------------------------------------------------
# Get unique names from the qb dataframe
unique_names = qb['NAME'].unique()

# Create a new dataframe to store the results
results = pd.DataFrame()

# Loop through each unique name
for name in unique_names:
    # Filter data for the current player
    player_data = qb[qb['NAME'] == name].sort_values('Year')

    # Create a copy of the player data for results
    player_result = player_data.copy()

    # Get number of data points for this player
    n_points = len(player_data)

    # Apply smoothing based on available data points
    try:
        if n_points >= 4:
            # Triple exponential smoothing for 4+ data points
            parvpg_series = player_data['PARVPG'].values
            model = ExponentialSmoothing(
                parvpg_series,
                trend='add',
                seasonal=None,
                damped_trend=True
            ).fit(smoothing_level=0.6, smoothing_trend=0.2, damping_trend=0.9)

            player_result['SmoothedPARVPG'] = model.fittedvalues
            print(f"Using triple exponential smoothing for {name} - {n_points} data points")

        elif n_points >= 2:
            # Double exponential smoothing for 2-3 data points
            parvpg_series = player_data['PARVPG'].values
            model = ExponentialSmoothing(
                parvpg_series,
                trend='add',
                seasonal=None
            ).fit()

            player_result['SmoothedPARVPG'] = model.fittedvalues
            print(f"Using double exponential smoothing for {name} - {n_points} data points")

        else:
            # Just use the original value for a single data point
            player_result['SmoothedPARVPG'] = player_data['PARVPG'].values
            print(f"Using original value for {name} - only 1 data point")

    except Exception as e:
        # Fallback to simple exponential smoothing if other methods fail
        print(f"Error processing {name} with primary method: {e}")

        try:
            if n_points >= 2:
                # Simple exponential smoothing as fallback
                parvpg_series = player_data['PARVPG'].values
                model = SimpleExpSmoothing(parvpg_series).fit(smoothing_level=0.6)
                player_result['SmoothedPARVPG'] = model.fittedvalues
                print(f"Fallback: Using simple exponential smoothing for {name}")
            else:
                # Just use the original value
                player_result['SmoothedPARVPG'] = player_data['PARVPG'].values
                print(f"Fallback: Using original value for {name} - only 1 data point")

        except Exception as e2:
            # Ultimate fallback - cumulative mean
            print(f"Second error for {name}: {e2}. Using cumulative mean.")
            player_result['SmoothedPARVPG'] = player_data['PARVPG'].expanding().mean().values

    # Append to results
    results = pd.concat([results, player_result])

# Sort the results by NAME and Year for better readability
results = results.sort_values(by=['NAME', 'Year'])
print(results.head(50)) #results arent right

