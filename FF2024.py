import os
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from typing import List, Union, Optional
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import QuantileTransformer

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
#print(qb.head())
print("check QB columns \n",qb.dtypes)

#start process of calculating PARV (Points Above Replacement Value)
qb['Rank'] = qb.groupby(['Year', 'WK'])['FPTS'].rank(method='dense', ascending=False)

# Step 1: Create parv_step with selected columns from qb
parv_step = qb[['Rank', 'Year', 'FPTS']].copy()

# Step 2: Filter for rows where Rank equals 12
parv_step = parv_step[parv_step['Rank'] == 12]

# Step 3: Group by Year and calculate average FPTS
avg_by_year = parv_step.groupby('Year')['FPTS'].mean().reset_index()
avg_by_year.rename(columns={'FPTS': 'Replacement Level Points'}, inplace=True)

print("check the result of avg by year")
print(avg_by_year.head())

#join in replacement level points to main QB dataframe
qb = qb.merge(avg_by_year, on='Year', how='inner')
qb['PARV']= qb['FPTS'] - qb['Replacement Level Points']
qb['PARV'] = np.where(qb['PARV'] < 0, 0, qb['PARV'])

tm_gather = qb.copy()
tm2= tm_gather.groupby(['NAME','Year'])['WK'].min().reset_index()
print("check tm2 \n", tm2.head())
tm_gather = pd.merge(tm_gather, tm2, on=['NAME','Year','WK'],how='inner')
tm_gather = tm_gather[['NAME','TEAM','Year']]
tm_gather['NAME'] = np.where(tm_gather['NAME'] =='Mitchell Trubisky','Mitch Trubisky',tm_gather['NAME'])

#print(qb.head())
#print(len(qb)," before Taysom filter")

#Taysom is an outlier, filter out
qb = qb[qb['NAME'] != 'Taysom Hill']
#print(len(qb))

#clean up name so it is consistent YOY
qb['NAME'] = np.where(qb['NAME'] =='Mitchell Trubisky','Mitch Trubisky',qb['NAME'])

qb = qb.groupby(['NAME','Year'])['PARV'].sum().reset_index()
#print(qb.head(), " after groupby")
#print(len(qb), " after groupby")

qb_games = read_files_to_dataframe(r'C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\QB results- FantasyPros')
#print(qb_games.dtypes)
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
#print(qb.head())

qb = qb[qb['G'] > 0]
qb['PARVPG'] = qb['PARV'] / qb['G']
qb = qb.sort_values(by=['NAME', 'Year'], ascending=[True, True])
qb_games.rename(columns={'G': 'Current Season Games Played','PARV':'Current_Season_PARV'}, inplace=True)


#PARVPG LOOP-----------------------------------------------------------------------------------------------------
# Get unique names from the qb dataframe
unique_names = qb['NAME'].unique()

# Define function to calculate RMSE between actual and forecast values
def calculate_rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast) ** 2))

# Function to optimize parameters for a given player's data
def optimize_params(player_data):
    n_points = len(player_data)

    if n_points <= 2:
        # Not enough data for meaningful optimization
        return 0.6, 0.2, 0.7  # Return default values

    # Function to minimize (RMSE)
    def objective_function(params):
        smoothing_level, smoothing_trend, damping_trend = params

        # Constrain parameters to sensible ranges
        if not (0 < smoothing_level < 1 and 0 < smoothing_trend < 1 and 0 < damping_trend < 1):
            return np.inf

        forecasts = np.zeros(n_points)
        # First year is always 0
        forecasts[0] = 0

        for i in range(1, n_points):
            prior_data = player_data.iloc[:i]['PARVPG'].values

            try:
                if len(prior_data) >= 2:  # Need at least 2 points for trend
                    model = ExponentialSmoothing(
                        prior_data,
                        trend='add',
                        seasonal=None,
                        damped_trend=True
                    ).fit(
                        smoothing_level=smoothing_level,
                        smoothing_trend=smoothing_trend,
                        damping_trend=damping_trend,
                        optimized=False
                    )
                    forecasts[i] = model.forecast(1)[0]
                else:
                    # For 2nd year, use simple smoothing
                    model = SimpleExpSmoothing(prior_data).fit(smoothing_level=smoothing_level, optimized=False)
                    forecasts[i] = model.forecast(1)[0]
            except:
                # In case of error, use mean of prior data
                forecasts[i] = np.mean(prior_data)

        # Calculate RMSE excluding the first point (which is always 0)
        rmse = calculate_rmse(player_data['PARVPG'].values[1:], forecasts[1:])
        return rmse

    # Initial parameter guesses
    initial_params = [0.6, 0.2, 0.7]

    # Optimize parameters
    bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]
    result = minimize(
        objective_function,
        initial_params,
        bounds=bounds,
        method='L-BFGS-B'
    )

    if result.success:
        return result.x
    else:
        print(f"Optimization failed: {result.message}")
        return initial_params


# Main processing loop
results = pd.DataFrame()  # Initialize empty results DataFrame

for name in unique_names:
    # Filter data for the current player
    player_data = qb[qb['NAME'] == name].sort_values('Year')

    # Create a copy of the player data for results
    player_result = player_data.copy()

    # Get number of data points for this player
    n_points = len(player_data)

    # Initialize SmoothedPARVPG column with zeros
    player_result['SmoothedPARVPG'] = 0.0

    # Optimize parameters if we have enough data points
    if n_points >= 3:
        print(f"Optimizing parameters for {name} with {n_points} data points...")
        optimal_smoothing_level, optimal_smoothing_trend, optimal_damping_trend = optimize_params(player_data)
        print(
            f"Optimal parameters for {name}: level={optimal_smoothing_level:.3f}, trend={optimal_smoothing_trend:.3f}, damping={optimal_damping_trend:.3f}")
    else:
        # Default parameters for players with limited data
        optimal_smoothing_level, optimal_smoothing_trend, optimal_damping_trend = 0.6, 0.2, 0.7
        print(f"Using default parameters for {name} (only {n_points} data points)")

    # Process each year for this player
    for i in range(n_points):
        current_year = player_data.iloc[i]['Year']

        # For the first year, SmoothedPARVPG is zero (already set)
        if i == 0:
            print(f"First year for {name} in {current_year}, setting SmoothedPARVPG to 0")
            continue

        # For subsequent years, use data from prior years only
        prior_data = player_data.iloc[:i]
        prior_parvpg = prior_data['PARVPG'].values

        # Apply smoothing based on available prior data points
        n_prior = len(prior_data)

        try:
            if n_prior >= 2:  # Need at least 2 prior points for trend
                # Exponential smoothing with optimized parameters
                model = ExponentialSmoothing(
                    prior_parvpg,
                    trend='add',
                    seasonal=None,
                    damped_trend=True
                ).fit(
                    smoothing_level=optimal_smoothing_level,
                    smoothing_trend=optimal_smoothing_trend,
                    damping_trend=optimal_damping_trend,
                    optimized=False
                )

                # Use the forecast for the next period (which is the current year)
                forecast = model.forecast(1)[0]
                player_result.iloc[i, player_result.columns.get_loc('SmoothedPARVPG')] = forecast
                print(f"Using exponential forecast for {name} - year {current_year} based on {n_prior} prior years")

            elif n_prior == 1:  # Only 1 prior point, use simple smoothing
                model = SimpleExpSmoothing(prior_parvpg).fit(smoothing_level=optimal_smoothing_level, optimized=False)
                forecast = model.forecast(1)[0]
                player_result.iloc[i, player_result.columns.get_loc('SmoothedPARVPG')] = forecast
                print(f"Using simple exponential forecast for {name} - year {current_year} with 1 prior year")

            else:
                # This shouldn't happen (we already handled i==0 case)
                print(f"Unexpected case for {name} - year {current_year}")

        except Exception as e:
            print(f"Error processing {name} for year {current_year}: {e}")

            try:
                # Fallback - use mean of prior values
                mean_value = np.mean(prior_parvpg)
                player_result.iloc[i, player_result.columns.get_loc('SmoothedPARVPG')] = mean_value
                print(f"Error for {name} - year {current_year}. Using mean of prior values: {mean_value:.4f}")

            except Exception as e2:
                print(f"Second error for {name} - year {current_year}: {e2}. Unable to process.")

    # Append to results
    results = pd.concat([results, player_result])

# Calculate overall RMSE for evaluation
valid_indices = (results.index > 0)  # Exclude first years (no division by zero concerns like MAPE)
overall_rmse = calculate_rmse(
    results.loc[valid_indices, 'PARVPG'].values,
    results.loc[valid_indices, 'SmoothedPARVPG'].values
)
print(f"Overall RMSE: {overall_rmse:.4f}")

# Sort the results by NAME and Year for better readability----------------------------------------------------------
results = results.sort_values(by=['NAME', 'Year'])
#print(results.head(50))

results['join name'] = results['NAME'].str.replace(' ', '').str.upper()
results = results[results['Year'] >= 2018]
#print("check length before team merge\n",len(results))
results = pd.merge(results, tm_gather,on=['NAME','Year'],how='inner')
#print("check length after team merge\n",len(results))
#print(results.head(50))


# start cleaning up Madden files for use-----------------------------------------------------------------------------
def extract_year_from_filename(filename):
    year_pattern = r'(20)\d{2}'
    match = re.search(year_pattern, filename)
    if match:
        return int(match.group())

    # Fallback: look for any 4-digit number
    four_digit_pattern = r'\d{4}'
    match = re.search(four_digit_pattern, filename)
    if match:
        year = int(match.group())
        # Reasonable year range check
        if 2006 <= year <= 2030:
            return year

    print(f"Warning: Could not extract year from filename: {filename}")
    return None


def standardize_column_names(df):
    # Column mapping dictionary
    column_mapping = {
        'Name': 'PLAYER',
        'name': 'PLAYER',
        'NAME': 'PLAYER',
        'Player': 'PLAYER',
        'player': 'PLAYER',
        'Player Name': 'PLAYER',
        'Full Name': 'PLAYER',

        'position': 'POS',
        'Position': 'POS',
        'POSITION': 'POS',
        'Pos': 'POS',
        'pos': 'POS',

        'team': 'TEAM',
        'Team': 'TEAM',
        'TEAM': 'TEAM',
        'tm': 'TEAM',
        'TM': 'TEAM',
        'Tm': 'TEAM',
        'Club': 'TEAM',
        'club': 'TEAM',
        'CLUB': 'TEAM',

        'OVR': 'Overall',
        'ovr': 'Overall',
        'overall': 'Overall',
        'OVERALL': 'Overall',
        'Overall Rating': 'Overall',

        'SPD': 'Speed',
        'spd': 'Speed',
        'speed': 'Speed',
        'SPEED': 'Speed',
        'Speed Rating': 'Speed',

        'AWR': 'Awareness',
        'awr': 'Awareness',
        'awareness': 'Awareness',
        'AWARENESS': 'Awareness',
        'Awareness Rating': 'Awareness',

        'Year': 'year',
        'YEAR': 'year',
        'Season': 'year',
        'season': 'year',
        'SEASON': 'year'
    }

    # Rename columns based on mapping
    df_renamed = df.rename(columns=column_mapping)
    return df_renamed


def clean_player_names(df):
    if 'PLAYER' in df.columns:
        df['PLAYER'] = df['PLAYER'].astype(str).str.replace(' ', '').str.upper()
    return df


def clean_team_names(df):
    if 'TEAM' in df.columns:
        df['TEAM'] = df['TEAM'].astype(str).str.strip().str.upper()
        # Replace common variations with standard abbreviations
        team_mapping = {
            'UNKNOWN': 'UNK',
            'FREE AGENT': 'FA',
            'FREEAGENT': 'FA',
            'N/A': 'FA',
            'NA': 'FA',
            '': 'FA'
        }
        df['TEAM'] = df['TEAM'].replace(team_mapping)
    return df


def ensure_column_types(df):
    # Define target types
    target_types = {
        'PLAYER': 'object',
        'POS': 'object',
        'TEAM': 'object',
        'Overall': 'int64',
        'Speed': 'int64',
        'Awareness': 'int64',
        'year': 'int64'
    }

    for col, target_type in target_types.items():
        if col in df.columns:
            try:
                if target_type == 'int64':
                    # Handle potential missing values and convert to int
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0).astype('int64')
                else:
                    df[col] = df[col].astype(target_type)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to {target_type}: {e}")

    return df


def process_single_excel_file(filepath):
    filename = os.path.basename(filepath)
    print(f"\nProcessing file: {filename}")

    try:
        # Try to read the Excel file
        # First, try to read without specifying sheet (will read first sheet)
        df = pd.read_excel(filepath)

        # Standardize column names
        df = standardize_column_names(df)
        print(f"After column standardization: {list(df.columns)}")

        # Add year column if not present
        if 'year' not in df.columns:
            year = extract_year_from_filename(filename)
            if year:
                df['year'] = year
                print(f"Added year column with value: {year}")
            else:
                print("Could not determine year - setting to 0")
                df['year'] = 0

        # Clean player names
        df = clean_player_names(df)

        # Clean team names
        df = clean_team_names(df)

        # Select only required columns (if they exist)
        required_columns = ['PLAYER', 'POS', 'TEAM', 'Overall', 'Speed', 'Awareness', 'year']
        available_columns = [col for col in required_columns if col in df.columns]

        if not available_columns:
            print(f"Warning: No required columns found in {filename}")
            return pd.DataFrame()

        # Filter to available columns
        df_filtered = df[available_columns].copy()

        # Add missing columns with default values
        for col in required_columns:
            if col not in df_filtered.columns:
                if col in ['PLAYER', 'POS']:
                    df_filtered[col] = 'UNKNOWN'
                elif col == 'TEAM':
                    df_filtered[col] = 'FA'  # Free Agent as default
                else:
                    df_filtered[col] = 0

        # Reorder columns
        df_filtered = df_filtered[required_columns]

        # Ensure correct data types
        df_filtered = ensure_column_types(df_filtered)

        # Remove rows with missing essential data
        df_filtered = df_filtered.dropna(subset=['PLAYER'])
        df_filtered = df_filtered[df_filtered['PLAYER'].str.strip() != '']
        df_filtered = df_filtered[df_filtered['PLAYER'] != 'UNKNOWN']

        print(f"Final shape after processing: {df_filtered.shape}")
        print(f"Sample of processed data:")
        print(df_filtered.head())

        return df_filtered

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return pd.DataFrame()


def combine_excel_files(file_paths):
    all_dataframes = []

    for filepath in file_paths:
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        df = process_single_excel_file(filepath)

        if not df.empty:
            all_dataframes.append(df)
        else:
            print(f"Skipping empty result from: {os.path.basename(filepath)}")

    if not all_dataframes:
        print("No valid data found in any files!")
        return pd.DataFrame()

    # Combine all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Final cleanup
    combined_df = combined_df.drop_duplicates()
    combined_df = combined_df.reset_index(drop=True)

    # Summary statistics
    print(f"\nSummary by year:")
    if 'year' in combined_df.columns:
        print(combined_df.groupby('year').size())

    print(f"\nSummary by position:")
    if 'POS' in combined_df.columns:
        print(combined_df['POS'].value_counts().head(10))

    print(f"\nSummary by team:")
    if 'TEAM' in combined_df.columns:
        print(combined_df['TEAM'].value_counts().head(10))

    return combined_df

excel_files = [
        r"C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\Madden data\maddennfl23fullplayerratings.xlsx",
        r"C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\Madden data\madden_nfl_22_player_ratings.xlsx",
        r"C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\Madden data\Madden 2021 Player Ratings.xlsx",
        r"C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\Madden data\madden_nfl_2020.xlsx",
        r"C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\Madden data\madden_nfl_2019.xlsx",
        r"C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\Madden data\madden_nfl_2018.xlsx"
    ]

final_df = combine_excel_files(excel_files)


final_df['PLAYER'] = np.where(final_df['PLAYER'] =='MITCHELLTRUBISKY','MITCHTRUBISKY',final_df['PLAYER'])
final_df = final_df.rename(columns={'PLAYER': 'join name','year': 'Year','TEAM': 'madden team'})
spt_staging = final_df.copy()
final_df = final_df[final_df['POS']=='QB']
#print("check final_df\n",final_df.head())
#print("check results df before merge\n",len(results))
qbdata = pd.merge(results, final_df, on=['join name','Year'],how='left')
#print("check qb data after merge\n",len(qbdata))

#check = qbdata[qbdata['_merge']=='left_only']
#print(check.head(100))
print("check qbdata columns\n",
      qbdata.info())

adp = read_files_to_dataframe(r"C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\ADP PPR data (FantasyData)",'.xlsx')
adp['Year'] = adp['source_file'].str[0:4].astype(int)
print("adp length before filters :",len(adp))
adp = adp[(adp['source_file'] !='2024 ADP data 2024-08-10.xlsx') & (adp['source_file'] !='2024 ADP data 2024-08-16.xlsx') & (adp['source_file'] !='2024 ADP data 2024-08-22.xlsx') & (adp['source_file'] !='2024 ADP data.xlsx')]
adp = adp.drop(columns=['RK','TEAM','BYE','AGE','POS RANK','POS RK','source_file'])
adp = adp[(adp['POS'] =='QB') | (adp['POS'] =='RB') | (adp['POS'] =='WR') | (adp['POS'] =='TE')]
adp = adp.drop(columns='POS')
print("adp length after filters :",len(adp))
print("check adp table")
print(adp.info())
print(adp.head(20))
adp['NAME'] = np.where(adp['NAME'] =='Mitchell Trubisky','Mitch Trubisky',adp['NAME'])
qbdata = pd.merge(qbdata, adp, on=['NAME','Year'],how='left')
qbdata['ADP'] = np.where(qbdata['ADP'] > 600, 600, qbdata['ADP'])

draft = read_files_to_dataframe(r'C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\draft data (Pro Football Reference)','.xlsx')
draft = draft.rename(columns={'Player': 'NAME'})
draft = draft[['NAME','Rnd','Pick']]
qbdata = pd.merge(qbdata, draft, on=['NAME'],how='left')
print("check qbdata columns after adp and draft merges\n",
      qbdata.info())


def calculate_supporting_cast_metrics(spt_staging): #---------------------------------------------------------------------
    # Filter for specific positions only
    target_positions = ['RT', 'C', 'LT', 'TE', 'RG', 'HB', 'LG', 'WR']
    filtered_df = spt_staging[spt_staging['POS'].isin(target_positions)].copy()

    print(f"Original dataframe shape: {spt_staging.shape}")
    print(f"After position filtering: {filtered_df.shape}")
    print(f"Positions included: {sorted(filtered_df['POS'].unique())}")

    # Initialize results list to store all metrics
    results = []

    # Group by Year and madden team
    for (year, team), group in filtered_df.groupby(['Year', 'madden team']):
        result_row = {
            'Year': year,
            'madden team': team
        }

        # Calculate Overall Supporting Cast
        # Top 1 for each position except WR (top 3)
        overall_support_values = []

        for pos in target_positions:
            pos_players = group[group['POS'] == pos].copy()
            if not pos_players.empty:
                if pos == 'WR':
                    # Top 3 WR
                    top_players = pos_players.nlargest(3, 'Overall')
                    overall_support_values.extend(top_players['Overall'].tolist())
                else:
                    # Top 1 for other positions
                    top_player = pos_players.nlargest(1, 'Overall')
                    overall_support_values.extend(top_player['Overall'].tolist())

        result_row['Overall Supporting Cast'] = np.mean(overall_support_values) if overall_support_values else np.nan

        # Calculate Offensive Supporting Cast (HB, TE, top 3 WR)
        offensive_positions = ['HB', 'TE', 'WR']
        offensive_support_values = []

        for pos in offensive_positions:
            pos_players = group[group['POS'] == pos].copy()
            if not pos_players.empty:
                if pos == 'WR':
                    # Top 3 WR
                    top_players = pos_players.nlargest(3, 'Overall')
                    offensive_support_values.extend(top_players['Overall'].tolist())
                else:
                    # Top 1 for HB and TE
                    top_player = pos_players.nlargest(1, 'Overall')
                    offensive_support_values.extend(top_player['Overall'].tolist())

        result_row['Offensive Supporting Cast'] = np.mean(
            offensive_support_values) if offensive_support_values else np.nan

        # Calculate Oline Support (RT, C, LT, RG, LG - top 1 each)
        oline_positions = ['RT', 'C', 'LT', 'RG', 'LG']
        oline_support_values = []

        for pos in oline_positions:
            pos_players = group[group['POS'] == pos].copy()
            if not pos_players.empty:
                top_player = pos_players.nlargest(1, 'Overall')
                oline_support_values.extend(top_player['Overall'].tolist())

        result_row['Oline Support'] = np.mean(oline_support_values) if oline_support_values else np.nan

        results.append(result_row)

    # Convert results to DataFrame
    metrics_df = pd.DataFrame(results)

    print(f"\nMetrics calculated for {len(metrics_df)} team-year combinations")
    print(f"Sample of calculated metrics:")
    print(metrics_df.head())

    # Join metrics back to original dataframe
    final_df = spt_staging.merge(
        metrics_df,
        on=['Year', 'madden team'],
        how='left'
    )

    print(f"\nFinal dataframe shape: {final_df.shape}")
    print(f"Columns in final dataframe: {list(final_df.columns)}")
    return final_df

spt_staging = calculate_supporting_cast_metrics(spt_staging)
spt_staging = spt_staging[['join name','Year','madden team', 'Overall Supporting Cast', 'Offensive Supporting Cast', 'Oline Support']]
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
#print("check spt_staging final results")
#print(spt_staging.head())
qbdata = pd.merge(qbdata, spt_staging, on=['join name','Year'],how='left')
qbdata = qbdata.drop(columns=['POS','madden team_x','madden team_y'])
print("check qbdata after spt merge, column drops")
qbdata.info()

s_grab = qbdata.groupby(['Year', 'TEAM'])[
    ['Overall Supporting Cast', 'Offensive Supporting Cast', 'Oline Support']
].max().reset_index()
#print("check s_grab\n",
 #     s_grab)
print(f"\nOriginal qbdata length: {len(qbdata)}")
qbdata = qbdata.merge(
    s_grab,
    on=['Year', 'TEAM'],
    how='inner',
    suffixes=('_original', '_max')
)
# Check merged data length
print(f"Merged data length: {len(qbdata)}")

# Combine original and max columns - use max values to fill null values in original columns
qbdata['Overall Supporting Cast'] = qbdata['Overall Supporting Cast_original'].fillna(
    qbdata['Overall Supporting Cast_max']
)

qbdata['Offensive Supporting Cast'] = qbdata['Offensive Supporting Cast_original'].fillna(
    qbdata['Offensive Supporting Cast_max']
)

qbdata['Oline Support'] = qbdata['Oline Support_original'].fillna(
    qbdata['Oline Support_max']
)

# Drop the temporary columns with suffixes
qbdata = qbdata.drop(columns=[
    'Overall Supporting Cast_original', 'Overall Supporting Cast_max',
    'Offensive Supporting Cast_original', 'Offensive Supporting Cast_max',
    'Oline Support_original', 'Oline Support_max','join name'
])
print("check qbdata after the supporting cast merger")
qbdata.info()


def fill_missing_madden_values(df):
    # Create a copy to avoid modifying the original dataframe
    df_filled = df.copy()

    # Ensure the dataframe is sorted by NAME and Year for proper forward fill
    df_filled = df_filled.sort_values(['NAME', 'Year'])

    # Columns to fill
    columns_to_fill = ['Overall', 'Speed', 'Awareness']

    # Group by NAME and forward fill the specified columns
    # This will use the previous year's value to fill missing values
    for col in columns_to_fill:
        if col in df_filled.columns:
            df_filled[col] = df_filled.groupby('NAME')[col].ffill()
        else:
            print(f"Warning: Column '{col}' not found in dataframe")

    return df_filled

qbdata = fill_missing_madden_values(qbdata)
#qbdata.to_excel(r'C:\Users\musel\OneDrive\Desktop\qb check.xlsx')

qbdata = qbdata.dropna(subset=['ADP', 'Overall'], how='all')
print(f"length after ADP & Overall filter: {len(qbdata)}")

#TRAIN TEST SPLIT----------------------------------------------------------------------------------------------------
qbdata['PARVPG_bins'] = pd.qcut(qbdata['PARVPG'], q=5, labels=False,duplicates='drop')
X = qbdata.drop(columns=['PARVPG'])  #features
y = qbdata['PARVPG']  #target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=qbdata['PARVPG_bins']
)
#Transform needed variables--------------------------------------------------------------------------------------
transformer = PowerTransformer(method='yeo-johnson', standardize=False)
columns_to_transform = ['SmoothedPARVPG', 'Pick']
# Fit ONLY on training data
X_train_transformed = X_train.copy()
X_train_transformed[columns_to_transform] = transformer.fit_transform(X_train[columns_to_transform])

# Apply the same transformation to test data (using parameters learned from training)
X_test_transformed = X_test.copy()
X_test_transformed[columns_to_transform] = transformer.transform(X_test[columns_to_transform])

qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_transformed['ADP_transformed'] = qt.fit_transform(X_train[['ADP']])
X_test_transformed['ADP_transformed'] = qt.transform(X_test[['ADP']])

y_train = transformer.fit_transform(y_train.values.reshape(-1, 1))
y_test = transformer.fit_transform(y_test.values.reshape(-1, 1))

print('check the distribution of the X data points')
X_train_transformed.hist(bins=7,figsize=(12,8))
plt.xticks(rotation=45)
plt.title("X_train_transformed")
plt.tight_layout()

print('check the distribution of the Y data points')
y_train.hist(bins=7,figsize=(12,8)) #need to fix this now
plt.xticks(rotation=45)
plt.title("y_train")
plt.tight_layout()

plt.show()

