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

        # Extract table headers - look for <small> tags which contain the actual column names
        thead = table.find('thead')
        if not thead:
            print("Could not find table header section")
            return None

        headers = []

        # First, try to find all <small> tags in the thead section
        small_tags = thead.find_all('small')
        if small_tags:
            print("Found <small> tags for headers")
            for small in small_tags:
                header_text = small.get_text(strip=True)
                if header_text:
                    headers.append(header_text)

        # If no <small> tags found, try the tablesorter-header approach
        if not headers:
            print("No <small> tags found, trying tablesorter-header approach")
            headers_row = thead.find('tr', class_='tablesorter-header')

            if not headers_row:
                # Fallback: try to find any tr with tablesorter-header elements
                all_rows = thead.find_all('tr')
                for row in all_rows:
                    if row.find('th', class_='tablesorter-header'):
                        headers_row = row
                        break

            if headers_row:
                for th in headers_row.find_all('th'):
                    # Look for the inner div with the actual header text
                    inner_div = th.find('div', class_='tablesorter-header-inner')
                    if inner_div:
                        header_text = inner_div.get_text(strip=True)
                    else:
                        header_text = th.get_text(strip=True)

                    if header_text:
                        headers.append(header_text)

        # Final fallback: try to extract from any th elements
        if not headers:
            print("Trying fallback header extraction from all th elements")
            all_th = thead.find_all('th')
            for th in all_th:
                # First try to find <small> tag within this th
                small = th.find('small')
                if small:
                    header_text = small.get_text(strip=True)
                elif 'Player' in th.get('class', []) or 'player' in str(th).lower():
                    header_text = "PLAYER"
                else:
                    header_text = th.get_text(strip=True)

                if header_text and header_text not in headers:
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
                # Handle player name cell which might have additional elements
                if 'player-name' in cell.get('class', []) or 'player' in str(cell.get('class', [])).lower():
                    # Extract just the player name, removing team info
                    player_link = cell.find('a')
                    if player_link:
                        player_name = player_link.get_text(strip=True)
                    else:
                        player_name = cell.get_text(strip=True)
                    row_data.append(player_name)
                else:
                    # For other cells, just get the text
                    cell_text = cell.get_text(strip=True)
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


def scrape_fantasy_projections2(url):
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

#data sources-----------------------------------------------------------------------------------------------------------
etr_values = pd.read_csv(r'C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\Temp_Rankings_Redraft_Auction 2025.csv')
etr_values['dollar diff'] = etr_values['ADP Yahoo'] - etr_values['ETR Half PPR']

wr_values = scrape_fantasy_projections2(r'https://www.fantasypros.com/nfl/projections/wr.php?filters=7493:152&week=draft&scoring=HALF&week=draft')
pd.set_option('display.max_columns', None)
print(wr_values.head())
#subset = wr_values[['RECEIVING','TDS']]
#print(subset.head())

