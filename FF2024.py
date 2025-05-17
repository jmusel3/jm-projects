import os
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
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

# Define function to calculate MAPE between actual and forecast values
def calculate_mape(actual, forecast):
    # Filter out where actual is 0 to avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return np.inf  # Return infinity if no valid points to calculate MAPE

    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100


# Function to optimize parameters for a given player's data
def optimize_params(player_data):
    n_points = len(player_data)

    if n_points <= 2:
        # Not enough data for meaningful optimization
        return 0.6, 0.2, 0.7  # Return default values

    # Function to minimize (MAPE)
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
                    model = SimpleExpSmoothing(prior_data).fit(smoothing_level=smoothing_level,optimized=False)
                    forecasts[i] = model.forecast(1)[0]
            except:
                # In case of error, use mean of prior data
                forecasts[i] = np.mean(prior_data)

        # Calculate MAPE excluding the first point (which is always 0)
        mape = calculate_mape(player_data['PARVPG'].values[1:], forecasts[1:])
        return mape

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
                model = SimpleExpSmoothing(prior_parvpg).fit(smoothing_level=optimal_smoothing_level,optimized=False)
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

# Calculate overall MAPE for evaluation
valid_indices = (results['PARVPG'] != 0) & (results.index > 0)  # Exclude first years
overall_mape = calculate_mape(
    results.loc[valid_indices, 'PARVPG'].values,
    results.loc[valid_indices, 'SmoothedPARVPG'].values
)
print(f"Overall MAPE: {overall_mape:.2f}%")



# Sort the results by NAME and Year for better readability
results = results.sort_values(by=['NAME', 'Year'])
print(results.head(50))

results['join name'] = results['NAME'].str.replace(' ', '').str.upper()

print(results.head(50))
