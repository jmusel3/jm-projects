#packages---------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from pulp import *
import itertools
from collections import defaultdict

#functions--------------------------------------------------------------------------------------------------------------
def scrape_fantasy_projections(url):
    """
    Scrapes fantasy football projections from FantasyPros website

    Args:
        url (str): The URL of the FantasyPros projections page

    Returns:
        pandas DataFrame with the projection data
    """

    # Headers to mimic a real browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Make the request
        print("Fetching data from FantasyPros...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the projections table
        table = soup.find('table', {'id': 'data'})

        if not table:
            print("Could not find the projections table. The page structure may have changed.")
            return None

        # Extract table headers - look for the actual column header row
        # Skip the first header row which contains category groupings like "RECEIVING", "RUSHING", "MISC"
        thead = table.find('thead')
        if not thead:
            print("Could not find table header section")
            return None

        headers = []
        category_keywords = ['RECEIVING', 'RUSHING', 'PASSING', 'MISC', 'KICKING', 'DEFENSE']

        # Look for the actual column header row (usually has tablesorter-header class)
        all_rows = thead.find_all('tr')
        column_header_row = None

        for row in all_rows:
            # Skip rows that contain category headers (have colspan attributes)
            has_colspan = any(cell.get('colspan') for cell in row.find_all(['td', 'th']))
            if has_colspan:
                continue  # Skip category header rows

            # This should be the actual column header row
            column_header_row = row
            break

        if not column_header_row:
            print("Could not find the column header row")
            return None

        # Extract headers from the column header row
        for cell in column_header_row.find_all(['th', 'td']):
            header_text = None

            # Method 1: Look for tablesorter-header-inner div (for Player column)
            inner_div = cell.find('div', class_='tablesorter-header-inner')
            if inner_div:
                header_text = inner_div.get_text(strip=True)

            # Method 2: Look for <small> tag (for stat columns)
            elif cell.find('small'):
                small_tag = cell.find('small')
                header_text = small_tag.get_text(strip=True)
                # Skip category headers
                if header_text.upper() in category_keywords:
                    continue

            # Method 3: Check if it's a player column by class
            elif 'player' in str(cell.get('class', [])).lower():
                header_text = "Player"

            # Method 4: Fallback to cell text
            else:
                header_text = cell.get_text(strip=True)

            if header_text and header_text.upper() not in category_keywords:
                headers.append(header_text)

        if not headers:
            print("Could not extract any headers from the table")
            return None

        print(f"Found headers: {headers}")

        # Extract table data
        tbody = table.find('tbody')
        if not tbody:
            print("Could not find table body")
            return None

        rows_data = []
        for row in tbody.find_all('tr'):
            row_data = []
            cells = row.find_all(['td', 'th'])

            for cell in cells:
                cell_text = ""

                # Handle player name cell which might have additional elements
                if 'player-name' in cell.get('class', []) or 'player' in str(cell.get('class', [])).lower():
                    # Extract just the player name, removing team info
                    player_link = cell.find('a')
                    if player_link:
                        cell_text = player_link.get_text(strip=True)
                    else:
                        cell_text = cell.get_text(strip=True)
                else:
                    # For other cells, get the text and clean it up
                    cell_text = cell.get_text(strip=True)

                    # Handle cases where numbers might have extra formatting
                    # Remove any non-digit, non-decimal characters for numeric cells
                    if cell_text and any(char.isdigit() for char in cell_text):
                        # Keep only digits, decimals, and negative signs
                        import re
                        numeric_match = re.search(r'-?\d*\.?\d+', cell_text)
                        if numeric_match:
                            cell_text = numeric_match.group()

                row_data.append(cell_text)

            if row_data:  # Only add non-empty rows
                rows_data.append(row_data)

        # Create DataFrame
        if rows_data:
            # Ensure all rows have the same number of columns as headers
            max_cols = len(headers)
            for i, row in enumerate(rows_data):
                if len(row) < max_cols:
                    rows_data[i].extend([''] * (max_cols - len(row)))
                elif len(row) > max_cols:
                    rows_data[i] = row[:max_cols]

            df = pd.DataFrame(rows_data, columns=headers)
            print(f"Successfully scraped {len(df)} fantasy football projections")
            return df
        else:
            print("No data rows found in the table")
            return None

    except requests.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return None
    except Exception as e:
        print(f"Error parsing the data: {e}")
        return None


def convert_player_names(df, name_column='Name'):
    """
    Convert player names in a DataFrame according to specific rules

    Args:
        df (DataFrame): The DataFrame containing player data
        name_column (str): The name of the column containing player names

    Returns:
        DataFrame: DataFrame with converted names
    """

    # Create a copy to avoid modifying the original DataFrame
    df_converted = df.copy()

    # Define specific name mappings
    name_mappings = {
        "DeVonta Smith": "Devonta Smith",
        "Tetairoa McMillan": "Tet McMillan",
        "Deebo Samuel Sr.": "Deebo Samuel",
        "Michael Pittman Jr.": "Michael Pittman",
        "Tre' Harris": "Tre Harris",
        "Patrick Mahomes II": "Patrick Mahomes",
        "Cameron Ward": "Cam Ward",
        "Anthony Richardson Sr.": "Anthony Richardson",
        "Aaron Jones Sr.": "Aaron Jones",
        "Brian Robinson Jr.": "Brian Robinson",
        "Travis Etienne Jr.": "Travis Etienne",
        "Kyle Pitts Sr.": "Kyle Pitts",
        "Arizona Cardinals": "ARI DST",
        "Atlanta Falcons": "ATL DST",
        "Baltimore Ravens": "BAL DST",
        "Buffalo Bills": "BUF DST",
        "Carolina Panthers": "CAR DST",
        "Chicago Bears": "CHI DST",
        "Cincinnati Bengals": "CIN DST",
        "Cleveland Browns": "CLE DST",
        "Dallas Cowboys": "DAL DST",
        "Denver Broncos": "DEN DST",
        "Detroit Lions": "DET DST",
        "Green Bay Packers": "GB DST",
        "Houston Texans": "HOU DST",
        "Indianapolis Colts": "IND DST",
        "Jacksonville Jaguars": "JAX DST",
        "Kansas City Chiefs": "KC DST",
        "Las Vegas Raiders": "LV DST",
        "Los Angeles Chargers": "LAC DST",
        "Los Angeles Rams": "LA DST",
        "Miami Dolphins": "MIA DST",
        "Minnesota Vikings": "MIN DST",
        "New England Patriots": "NE DST",
        "New Orleans Saints": "NO DST",
        "New York Giants": "NYG DST",
        "New York Jets": "NYJ DST",
        "Philadelphia Eagles": "PHI DST",
        "Pittsburgh Steelers": "PIT DST",
        "San Francisco 49ers": "SF DST",
        "Seattle Seahawks": "SEA DST",
        "Tampa Bay Buccaneers": "TB DST",
        "Tennessee Titans": "TEN DST",
        "Washington Commanders": "WAS DST"
    }

    # Apply specific name mappings
    df_converted[name_column] = df_converted[name_column].replace(name_mappings)
    return df_converted


class FantasyFootballOptimizer:
    def __init__(self, auction_values_df):
        """
        Initialize the optimizer with auction values dataframe

        Expected columns:
        - 'ETR Half PPR': auction dollar values
        - 'FPTS': fantasy points
        - 'Position': player position
        """
        self.df = auction_values_df.copy()
        self.budget = 200
        self.scenarios = []

        # Clean and prepare data
        self.prepare_data()

    def prepare_data(self):
        """Clean and prepare the data for optimization"""
        # Remove any rows with missing values in key columns
        self.df = self.df.dropna(subset=['ETR Half PPR', 'FPTS', 'Position'])

        # Ensure numeric columns are numeric
        self.df['ETR Half PPR'] = pd.to_numeric(self.df['ETR Half PPR'], errors='coerce')
        self.df['FPTS'] = pd.to_numeric(self.df['FPTS'], errors='coerce')

        # Remove any remaining NaN values
        self.df = self.df.dropna(subset=['ETR Half PPR', 'FPTS'])

        # Add player index
        self.df.reset_index(drop=True, inplace=True)
        self.df['player_idx'] = self.df.index

        # Group positions for flex eligibility
        self.flex_positions = ['RB', 'WR', 'TE']

    def get_position_requirements(self):
        """Define roster requirements"""
        return {
            'QB': {'min': 1, 'max': 1},  # max 2 only if top 7 not drafted
            'RB': {'min': 2, 'max': 6},
            'WR': {'min': 3, 'max': 6},
            'TE': {'min': 1, 'max': 2},  # max 2 only if top 7 not drafted
            'DST': {'min': 1, 'max': 1},
            'K': {'min': 0, 'max': 15},
            'flex': {'min': 1, 'max': 1},  # Additional RB/WR/TE
            'total': {'min': 15, 'max': 15}  # Including bench
        }

    def create_optimization_problem(self, exclude_players=None):
        """Create and solve the optimization problem"""
        if exclude_players is None:
            exclude_players = set()

        # Filter out excluded players
        available_df = self.df[~self.df['player_idx'].isin(exclude_players)].copy()

        # Create binary variables for each player
        player_vars = {}
        starter_vars = {}  # Variables to identify starters

        for idx, row in available_df.iterrows():
            player_vars[row['player_idx']] = LpVariable(f"player_{row['player_idx']}", cat='Binary')

        # Create starter variables for each starting position
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
            pos_players = available_df[available_df['Position'] == pos]['player_idx'].tolist()
            for player_idx in pos_players:
                starter_vars[f"{pos}_{player_idx}"] = LpVariable(f"starter_{pos}_{player_idx}", cat='Binary')

        # Flex starters (additional RB/WR/TE)
        flex_players = available_df[available_df['Position'].isin(self.flex_positions)]['player_idx'].tolist()
        for player_idx in flex_players:
            starter_vars[f"FLEX_{player_idx}"] = LpVariable(f"starter_FLEX_{player_idx}", cat='Binary')

        # Create the problem
        prob = LpProblem("Fantasy_Football_Auction", LpMaximize)

        # Objective: Heavily weight starters (10x) vs bench players (1x)
        starter_weight = 10.0
        bench_weight = 1.0

        starter_fpts = lpSum([
            available_df.loc[available_df['player_idx'] == int(var.name.split('_')[-1]), 'FPTS'].iloc[
                0] * var * starter_weight
            for var in starter_vars.values()
        ])

        total_fpts = lpSum([
            available_df.loc[available_df['player_idx'] == player_idx, 'FPTS'].iloc[0] * var * bench_weight
            for player_idx, var in player_vars.items()
        ])

        prob += starter_fpts + total_fpts

        # Budget constraint
        prob += lpSum([
            available_df.loc[available_df['player_idx'] == player_idx, 'ETR Half PPR'].iloc[0] * var
            for player_idx, var in player_vars.items()
        ]) <= self.budget

        # Starter constraints - link starter variables to player variables
        # QB starters (exactly 1)
        qb_players = available_df[available_df['Position'] == 'QB']['player_idx'].tolist()
        qb_starter_vars = [starter_vars[f"QB_{p}"] for p in qb_players if f"QB_{p}" in starter_vars]
        if qb_starter_vars:
            prob += lpSum(qb_starter_vars) == 1
            for p in qb_players:
                if f"QB_{p}" in starter_vars:
                    # Starter can only be selected if player is on roster
                    prob += starter_vars[f"QB_{p}"] <= player_vars[p]

        # RB starters (exactly 2)
        rb_players = available_df[available_df['Position'] == 'RB']['player_idx'].tolist()
        rb_starter_vars = [starter_vars[f"RB_{p}"] for p in rb_players if f"RB_{p}" in starter_vars]
        if rb_starter_vars:
            prob += lpSum(rb_starter_vars) == 2
            for p in rb_players:
                if f"RB_{p}" in starter_vars:
                    prob += starter_vars[f"RB_{p}"] <= player_vars[p]

        # WR starters (exactly 3)
        wr_players = available_df[available_df['Position'] == 'WR']['player_idx'].tolist()
        wr_starter_vars = [starter_vars[f"WR_{p}"] for p in wr_players if f"WR_{p}" in starter_vars]
        if wr_starter_vars:
            prob += lpSum(wr_starter_vars) == 3
            for p in wr_players:
                if f"WR_{p}" in starter_vars:
                    prob += starter_vars[f"WR_{p}"] <= player_vars[p]

        # TE starters (exactly 1)
        te_players = available_df[available_df['Position'] == 'TE']['player_idx'].tolist()
        te_starter_vars = [starter_vars[f"TE_{p}"] for p in te_players if f"TE_{p}" in starter_vars]
        if te_starter_vars:
            prob += lpSum(te_starter_vars) == 1
            for p in te_players:
                if f"TE_{p}" in starter_vars:
                    prob += starter_vars[f"TE_{p}"] <= player_vars[p]

        # DST starters (exactly 1)
        dst_players = available_df[available_df['Position'] == 'DST']['player_idx'].tolist()
        dst_starter_vars = [starter_vars[f"DST_{p}"] for p in dst_players if f"DST_{p}" in starter_vars]
        if dst_starter_vars:
            prob += lpSum(dst_starter_vars) == 1
            for p in dst_players:
                if f"DST_{p}" in starter_vars:
                    prob += starter_vars[f"DST_{p}"] <= player_vars[p]

        # Flex starter (exactly 1 additional RB/WR/TE)
        flex_starter_vars = [starter_vars[f"FLEX_{p}"] for p in flex_players if f"FLEX_{p}" in starter_vars]
        if flex_starter_vars:
            prob += lpSum(flex_starter_vars) == 1
            for p in flex_players:
                if f"FLEX_{p}" in starter_vars:
                    prob += starter_vars[f"FLEX_{p}"] <= player_vars[p]
                    # Player can't be both position starter AND flex starter
                    if p in rb_players and f"RB_{p}" in starter_vars:
                        prob += starter_vars[f"FLEX_{p}"] + starter_vars[f"RB_{p}"] <= 1
                    if p in wr_players and f"WR_{p}" in starter_vars:
                        prob += starter_vars[f"FLEX_{p}"] + starter_vars[f"WR_{p}"] <= 1
                    if p in te_players and f"TE_{p}" in starter_vars:
                        prob += starter_vars[f"FLEX_{p}"] + starter_vars[f"TE_{p}"] <= 1

        # Position constraints for total roster
        positions = available_df['Position'].unique()
        requirements = self.get_position_requirements()

        for pos in positions:
            if pos in requirements:
                pos_players = available_df[available_df['Position'] == pos]['player_idx'].tolist()
                pos_vars = [player_vars[p] for p in pos_players if p in player_vars]

                if pos_vars:
                    # Minimum requirement (handled by starter constraints above)
                    # Maximum constraints
                    if pos in ['QB', 'TE']:
                        # Special logic for QB and TE (max 2 only if top 7 not drafted)
                        top_7_pos = available_df[available_df['Position'] == pos].nlargest(7, 'FPTS')[
                            'player_idx'].tolist()
                        top_7_vars = [player_vars[p] for p in top_7_pos if p in player_vars]

                        if top_7_vars:
                            # Create binary variable to track if any top 7 are selected
                            top7_selected = LpVariable(f"top7_{pos}_selected", cat='Binary')
                            prob += lpSum(top_7_vars) >= top7_selected
                            prob += lpSum(top_7_vars) <= len(top_7_vars) * top7_selected

                            # If top 7 selected, allow max 2, otherwise max 1
                            prob += lpSum(pos_vars) <= 1 + top7_selected
                        else:
                            prob += lpSum(pos_vars) <= 1
                    elif pos == 'DST':
                        prob += lpSum(pos_vars) == 1  # Exactly 1 DST
                    else:
                        prob += lpSum(pos_vars) <= requirements[pos]['max']

        # Total roster size
        prob += lpSum(player_vars.values()) == requirements['total']['min']

        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=0))

        if prob.status == 1:  # Optimal solution found
            selected_players = []
            starters = []

            # Identify selected players and starters
            for player_idx, var in player_vars.items():
                if var.varValue == 1:
                    player_info = available_df[available_df['player_idx'] == player_idx].iloc[0]
                    player_data = {
                        'player_idx': player_idx,
                        'name': player_info.get('Name', f'Player_{player_idx}'),
                        'position': player_info['Position'],
                        'cost': player_info['ETR Half PPR'],
                        'fpts': player_info['FPTS'],
                        'is_starter': False
                    }

                    # Check if this player is a starter
                    for starter_key, starter_var in starter_vars.items():
                        if starter_var.varValue == 1 and str(player_idx) in starter_key:
                            player_data['is_starter'] = True
                            player_data['starter_position'] = starter_key.split('_')[0]
                            break

                    selected_players.append(player_data)

            return selected_players, prob.objective.value()
        else:
            return None, None

    def find_backup_players(self, selected_players):
        """Find backup players for each position with similar dollar values"""
        backups = {}
        selected_indices = {p['player_idx'] for p in selected_players}

        # Group selected players by position and find their average cost
        position_costs = defaultdict(list)
        for player in selected_players:
            position_costs[player['position']].append(player['cost'])

        # Calculate average cost per position
        avg_costs = {}
        for pos, costs in position_costs.items():
            avg_costs[pos] = sum(costs) / len(costs)

        # Find backups for each position with similar cost
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
            if pos in avg_costs:
                target_cost = avg_costs[pos]

                # Get available players at this position
                pos_df = self.df[
                    (self.df['Position'] == pos) &
                    (~self.df['player_idx'].isin(selected_indices))
                    ].copy()

                if not pos_df.empty:
                    # Calculate cost difference from target and sort by smallest difference
                    pos_df['cost_diff'] = abs(pos_df['ETR Half PPR'] - target_cost)

                    # First, try to find players within 20% of target cost
                    cost_tolerance = target_cost * 0.2
                    similar_cost_df = pos_df[pos_df['cost_diff'] <= cost_tolerance]

                    if not similar_cost_df.empty:
                        # Among similar cost players, pick the one with highest FPTS
                        backup = similar_cost_df.nlargest(1, 'FPTS').iloc[0]
                    else:
                        # If no players within 20%, pick closest cost with decent FPTS
                        # Sort by cost difference, then by FPTS descending
                        pos_df = pos_df.sort_values(['cost_diff', 'FPTS'], ascending=[True, False])
                        backup = pos_df.iloc[0]

                    backups[pos] = {
                        'name': backup.get('Name', f'Player_{backup["player_idx"]}'),
                        'position': backup['Position'],
                        'cost': backup['ETR Half PPR'],
                        'fpts': backup['FPTS'],
                        'cost_diff_from_selected': abs(backup['ETR Half PPR'] - target_cost)
                    }

        return backups

    def generate_scenarios(self, num_scenarios=6):
        """Generate multiple optimal scenarios with completely different player combinations"""
        scenarios = []
        all_excluded_players = set()

        for scenario_num in range(num_scenarios):
            selected_players, total_fpts = self.create_optimization_problem(all_excluded_players)

            if selected_players:
                # Find backup players
                backups = self.find_backup_players(selected_players)

                # Calculate totals
                total_cost = sum(p['cost'] for p in selected_players)
                starter_fpts = sum(p['fpts'] for p in selected_players if p['is_starter'])
                bench_fpts = sum(p['fpts'] for p in selected_players if not p['is_starter'])

                scenarios.append({
                    'scenario_num': scenario_num + 1,
                    'players': selected_players,
                    'backups': backups,
                    'total_cost': total_cost,
                    'total_fpts': sum(p['fpts'] for p in selected_players),
                    'starter_fpts': starter_fpts,
                    'bench_fpts': bench_fpts,
                    'remaining_budget': self.budget - total_cost
                })

                # Exclude ALL players from this scenario for future scenarios
                # This ensures completely different rosters
                current_players = {p['player_idx'] for p in selected_players}
                all_excluded_players.update(current_players)

            if len(scenarios) >= num_scenarios:
                break

        return scenarios

    def print_scenarios(self, scenarios):
        """Print formatted scenarios"""
        for scenario in scenarios:
            print(f"\n{'=' * 60}")
            print(f"SCENARIO {scenario['scenario_num']}")
            print(f"{'=' * 60}")
            print(f"Total Cost: ${scenario['total_cost']:.1f}")
            print(f"Starting Lineup FPTS: {scenario['starter_fpts']:.1f}")
            print(f"Bench FPTS: {scenario['bench_fpts']:.1f}")
            print(f"Total FPTS: {scenario['total_fpts']:.1f}")
            print(f"Remaining Budget: ${scenario['remaining_budget']:.1f}")

            # Separate starters and bench
            starters = [p for p in scenario['players'] if p['is_starter']]
            bench = [p for p in scenario['players'] if not p['is_starter']]

            print(f"\nSTARTING LINEUP:")
            print("-" * 40)

            # Group starters by their starting position
            starter_positions = ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST']
            starters_by_pos = defaultdict(list)

            for player in starters:
                starter_pos = player.get('starter_position', player['position'])
                starters_by_pos[starter_pos].append(player)

            for pos in starter_positions:
                if pos in starters_by_pos:
                    if pos == 'RB' and len(starters_by_pos[pos]) == 2:
                        print(
                            f"RB1: {starters_by_pos[pos][0]['name']} - ${starters_by_pos[pos][0]['cost']:.1f} ({starters_by_pos[pos][0]['fpts']:.1f} pts)")
                        print(
                            f"RB2: {starters_by_pos[pos][1]['name']} - ${starters_by_pos[pos][1]['cost']:.1f} ({starters_by_pos[pos][1]['fpts']:.1f} pts)")
                    elif pos == 'WR' and len(starters_by_pos[pos]) == 3:
                        for i, player in enumerate(starters_by_pos[pos], 1):
                            print(f"WR{i}: {player['name']} - ${player['cost']:.1f} ({player['fpts']:.1f} pts)")
                    else:
                        for player in starters_by_pos[pos]:
                            print(f"{pos}: {player['name']} - ${player['cost']:.1f} ({player['fpts']:.1f} pts)")

            if bench:
                print(f"\nBENCH:")
                print("-" * 40)
                bench_by_pos = defaultdict(list)
                for player in bench:
                    bench_by_pos[player['position']].append(player)

                for pos in ['QB', 'RB', 'WR', 'TE', 'DST', 'K']:
                    if pos in bench_by_pos:
                        for player in bench_by_pos[pos]:
                            print(f"{pos}: {player['name']} - ${player['cost']:.1f} ({player['fpts']:.1f} pts)")

            print(f"\nBACKUP OPTIONS:")
            print("-" * 40)
            for pos, backup in scenario['backups'].items():
                cost_diff = backup.get('cost_diff_from_selected', 0)
                print(f"{pos}: {backup['name']} - ${backup['cost']:.1f} ({backup['fpts']:.1f} pts) [±${cost_diff:.1f}]")

#data sources-----------------------------------------------------------------------------------------------------------
etr_values = pd.read_csv(r'C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\Temp_Rankings_Redraft_Auction 2025.csv')
etr_values = etr_values[(etr_values['Position'] != 'K')]
etr_values['dollar diff'] = etr_values['ADP Yahoo'] - etr_values['ETR Half PPR']

wr_values = scrape_fantasy_projections(r'https://www.fantasypros.com/nfl/projections/wr.php?filters=7493:152&week=draft&scoring=HALF&week=draft')
pd.set_option('display.max_columns', None)
wr_values = wr_values[['Player','FPTS']]

qb_values = scrape_fantasy_projections(r'https://www.fantasypros.com/nfl/projections/qb.php?filters=7493:152&week=draft')
qb_values = qb_values[['Player','FPTS']]


rb_values = scrape_fantasy_projections(r'https://www.fantasypros.com/nfl/projections/rb.php?filters=7493:152&week=draft&scoring=HALF')
rb_values = rb_values[['Player','FPTS']]

te_values = scrape_fantasy_projections(r'https://www.fantasypros.com/nfl/projections/te.php?filters=7493:152&week=draft&scoring=HALF&week=draft')
te_values = te_values[['Player','FPTS']]

dst_values = scrape_fantasy_projections(r'https://www.fantasypros.com/nfl/projections/dst.php?filters=7493:152&week=draft')
dst_values = dst_values[['Player','FPTS']]

#combine projections across positions-----------------------------------------------------------------------------------
player_values = pd.concat([qb_values, rb_values, wr_values, te_values, dst_values],axis=0)
player_values = player_values.rename(columns={'Player': 'Name'})
player_values = convert_player_names(player_values)

#join the projections to auction values---------------------------------------------------------------------------------
auction_values = pd.merge(etr_values, player_values, on=['Name'],how='left',indicator=True)
data_check = auction_values[(auction_values['_merge']=='left_only')]
data_check.to_csv(r'C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\name data checking.csv')
auction_values['FPTS'] = auction_values['FPTS'].fillna(0)
auction_values['ETR Half PPR'] = np.where(auction_values['ETR Half PPR'] < auction_values['ADP Yahoo'], auction_values['ADP Yahoo'],auction_values['ETR Half PPR'])

#run optimizer----------------------------------------------------------------------------------------------------------
optimizer = FantasyFootballOptimizer(auction_values)
scenarios = optimizer.generate_scenarios(6)
optimizer.print_scenarios(scenarios)