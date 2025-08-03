#packages---------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

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