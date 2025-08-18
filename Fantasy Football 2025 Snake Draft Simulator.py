import pandas as pd
import numpy as np
from copy import deepcopy
import itertools


class FantasyFootballDraftSimulator:
    def __init__(self, csv_path):
        """Initialize the draft simulator with player data"""
        self.df = pd.read_csv(csv_path)
        self.num_teams = 10
        self.rounds = 16  # Total roster spots

        # Required starting positions
        self.required_positions = {
            'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1,
            'FLEX': 1, 'DST': 1, 'K': 1
        }

        # Position limits
        self.position_limits = {
            'QB': 2, 'RB': 7, 'WR': 6, 'TE': 2, 'DST': 1, 'K': 1
        }

        # QB/TE thresholds for second pick
        self.qb_threshold = 5
        self.te_threshold = 7

        # Pre-assigned players with their keeper round values
        self.pre_assigned = {
            'Jared': ('Brian Thomas Jr.', 8),
            'Mike': ("Ja'Marr Chase", 5),
            'Mikey': ('Ladd McConkey', 8),
            'Simon': ('Bucky Irving', 8),
            'Litty': ('Lamar Jackson', 5),
            'Cub': ('Trey McBride', 7),
            'Luke': ('Garrett Wilson', 6),
            'Matt': ("De'Von Achane", 2),
            'Forrest': ('Puka Nacua', 7),
            'Jake': ('Chase Brown', 8)
        }

        self.team_names = list(self.pre_assigned.keys())

        # Clean and prepare data
        self.prepare_data()

    def prepare_data(self):
        """Clean and prepare the player data"""
        # Remove any rows with missing values in key columns
        self.df = self.df.dropna(subset=['Name', 'Pos', 'ETR Full PPR'])

        # Sort by ETR Full PPR value (descending)
        self.df = self.df.sort_values('ETR Full PPR', ascending=False).reset_index(drop=True)

        # Create position rankings
        self.position_rankings = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST', 'K']:
            pos_df = self.df[self.df['Pos'] == pos].reset_index(drop=True)
            self.position_rankings[pos] = pos_df

    def get_snake_pick_order(self, draft_position, round_num):
        """Get the actual pick number in a snake draft"""
        if round_num % 2 == 1:  # Odd rounds (1, 3, 5, ...)
            return (round_num - 1) * self.num_teams + draft_position
        else:  # Even rounds (2, 4, 6, ...)
            return (round_num - 1) * self.num_teams + (self.num_teams - draft_position + 1)

    def get_team_picking(self, overall_pick):
        """Get which team is picking at a given overall pick"""
        round_num = ((overall_pick - 1) // self.num_teams) + 1
        position_in_round = ((overall_pick - 1) % self.num_teams) + 1

        if round_num % 2 == 1:  # Odd rounds
            return position_in_round
        else:  # Even rounds
            return self.num_teams - position_in_round + 1

    def calculate_positional_scarcity(self, position, available_players, teams_state):
        """Calculate positional scarcity score based on remaining value drop-off"""
        if position not in available_players or len(available_players[position]) == 0:
            return 0

        pos_players = available_players[position]
        if len(pos_players) < 2:
            return pos_players.iloc[0]['ETR Full PPR'] if len(pos_players) > 0 else 0

        # Calculate value drop from best to 5th best remaining
        top_value = pos_players.iloc[0]['ETR Full PPR']
        fifth_value = pos_players.iloc[min(4, len(pos_players) - 1)]['ETR Full PPR']

        return max(0, top_value - fifth_value)

    def evaluate_team_needs(self, team_roster, available_players):
        """Evaluate what positions a team most needs"""
        position_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DST': 0, 'K': 0}

        for player in team_roster:
            pos = player['Pos']
            if pos in position_counts:
                position_counts[pos] += 1

        needs = {}

        # Calculate need scores based on requirements and limits
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST', 'K']:
            required = self.required_positions.get(pos, 0)
            if pos == 'RB' or pos == 'WR' or pos == 'TE':
                required += 1  # Account for FLEX

            current = position_counts[pos]
            limit = self.position_limits[pos]

            if current < required:
                needs[pos] = 10  # High need
            elif current < limit:
                # Special rules for QB and TE
                if pos == 'QB' and current >= 1:
                    qb_taken = len(self.position_rankings['QB']) - len(available_players.get('QB', []))
                    needs[pos] = 3 if qb_taken < self.qb_threshold else 0
                elif pos == 'TE' and current >= 1:
                    te_taken = len(self.position_rankings['TE']) - len(available_players.get('TE', []))
                    needs[pos] = 2 if te_taken < self.te_threshold else 0
                else:
                    needs[pos] = 5  # Medium need
            else:
                needs[pos] = 0  # No need

        return needs

    def select_best_player(self, available_players, team_roster):
        """Select the best available player considering team needs and scarcity"""
        if not available_players:
            return None

        team_needs = self.evaluate_team_needs(team_roster, available_players)
        best_player = None
        best_score = -float('inf')

        for pos in available_players:
            if len(available_players[pos]) == 0 or team_needs.get(pos, 0) == 0:
                continue

            top_player = available_players[pos].iloc[0]

            # Base value
            player_value = top_player['ETR Full PPR']

            # Need multiplier
            need_score = team_needs[pos]

            # Positional scarcity
            scarcity_score = self.calculate_positional_scarcity(pos, available_players, team_roster)

            # Combined score
            total_score = player_value + (need_score * 2) + (scarcity_score * 0.1)

            if total_score > best_score:
                best_score = total_score
                best_player = top_player.to_dict()

        return best_player

    def simulate_draft(self, jared_position):
        """Simulate a complete draft with Jared in the specified position"""
        # Initialize team rosters
        teams = {i: [] for i in range(1, self.num_teams + 1)}

        # Create available players dict by position
        available_players = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST', 'K']:
            available_players[pos] = self.position_rankings[pos].copy()

        # Track pre-assigned players and their rounds
        pre_assigned_info = {}
        for i, team_name in enumerate(self.team_names, 1):
            if team_name in self.pre_assigned:
                player_name, keeper_round = self.pre_assigned[team_name]
                pre_assigned_info[i] = (player_name, keeper_round)

        # Assign Jared to the specified position
        jared_team = jared_position
        if 'Jared' in self.pre_assigned:
            pre_assigned_info[jared_team] = self.pre_assigned['Jared']

        # Assign other teams to remaining positions
        remaining_positions = [i for i in range(1, self.num_teams + 1) if i != jared_position]
        other_teams = [name for name in self.team_names if name != 'Jared']

        for i, team_name in enumerate(other_teams):
            if i < len(remaining_positions) and team_name in self.pre_assigned:
                pre_assigned_info[remaining_positions[i]] = self.pre_assigned[team_name]

        # Remove pre-assigned players from available players
        for team_pos, (player_name, keeper_round) in pre_assigned_info.items():
            for pos in available_players:
                available_players[pos] = available_players[pos][
                    available_players[pos]['Name'] != player_name
                    ].reset_index(drop=True)

        # Conduct the draft
        for overall_pick in range(1, self.num_teams * self.rounds + 1):
            team_picking = self.get_team_picking(overall_pick)
            current_round = ((overall_pick - 1) // self.num_teams) + 1

            # Check if this team has a pre-assigned player for this round
            if (team_picking in pre_assigned_info and
                    pre_assigned_info[team_picking][1] == current_round):

                player_name, _ = pre_assigned_info[team_picking]
                # Find the player in original dataframe
                player_row = self.df[self.df['Name'] == player_name].iloc[0]
                selected_player = player_row.to_dict()

            else:
                # Select best available player
                selected_player = self.select_best_player(available_players, teams[team_picking])

                if selected_player:
                    # Remove selected player from available players
                    pos = selected_player['Pos']
                    available_players[pos] = available_players[pos][
                        available_players[pos]['Name'] != selected_player['Name']
                        ].reset_index(drop=True)

            if selected_player:
                teams[team_picking].append(selected_player)

        return teams

    def calculate_team_score(self, roster):
        """Calculate total team score based on starting lineup requirements"""
        if not roster:
            return 0

        # Separate players by position
        by_position = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'DST': [], 'K': []}

        for player in roster:
            pos = player['Pos']
            if pos in by_position:
                by_position[pos].append(player)

        # Sort each position by ETR Full PPR value
        for pos in by_position:
            by_position[pos].sort(key=lambda x: x['ETR Full PPR'], reverse=True)

        total_score = 0

        # Add required starters
        total_score += sum(p['ETR Full PPR'] for p in by_position['QB'][:1])
        total_score += sum(p['ETR Full PPR'] for p in by_position['RB'][:2])
        total_score += sum(p['ETR Full PPR'] for p in by_position['WR'][:3])
        total_score += sum(p['ETR Full PPR'] for p in by_position['TE'][:1])
        total_score += sum(p['ETR Full PPR'] for p in by_position['DST'][:1])
        total_score += sum(p['ETR Full PPR'] for p in by_position['K'][:1])

        # Add best FLEX player (remaining RB/WR/TE)
        flex_candidates = []
        flex_candidates.extend(by_position['RB'][2:])  # RBs beyond RB1, RB2
        flex_candidates.extend(by_position['WR'][3:])  # WRs beyond WR1, WR2, WR3
        flex_candidates.extend(by_position['TE'][1:])  # TEs beyond TE1

        if flex_candidates:
            best_flex = max(flex_candidates, key=lambda x: x['ETR Full PPR'])
            total_score += best_flex['ETR Full PPR']

        return total_score

    def run_all_simulations(self):
        """Run draft simulations for all possible Jared positions"""
        results = {}

        print("Running draft simulations for all positions...")

        for position in range(1, self.num_teams + 1):
            print(f"Simulating draft with Jared in position {position}...")

            teams = self.simulate_draft(position)
            jared_roster = teams[position]
            jared_score = self.calculate_team_score(jared_roster)

            results[position] = {
                'score': jared_score,
                'roster': jared_roster
            }

        return results

    def display_results(self, results):
        """Display the results ranked by best draft position"""
        print("\n" + "=" * 60)
        print("DRAFT POSITION RANKINGS FOR JARED")
        print("=" * 60)

        # Sort by score (descending)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)

        for rank, (position, data) in enumerate(sorted_results, 1):
            print(f"\nRank {rank}: Draft Position {position}")
            print(f"Total Starting Lineup Score: {data['score']:.2f}")

            # Show starting lineup
            roster = data['roster']
            by_position = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'DST': [], 'K': []}

            for player in roster:
                pos = player['Pos']
                if pos in by_position:
                    by_position[pos].append(player)

            for pos in by_position:
                by_position[pos].sort(key=lambda x: x['ETR Full PPR'], reverse=True)

            print("\nStarting Lineup:")
            print(f"QB: {by_position['QB'][0]['Name'] if by_position['QB'] else 'None'}")
            print(f"RB1: {by_position['RB'][0]['Name'] if len(by_position['RB']) > 0 else 'None'}")
            print(f"RB2: {by_position['RB'][1]['Name'] if len(by_position['RB']) > 1 else 'None'}")
            print(f"WR1: {by_position['WR'][0]['Name'] if len(by_position['WR']) > 0 else 'None'}")
            print(f"WR2: {by_position['WR'][1]['Name'] if len(by_position['WR']) > 1 else 'None'}")
            print(f"WR3: {by_position['WR'][2]['Name'] if len(by_position['WR']) > 2 else 'None'}")
            print(f"TE: {by_position['TE'][0]['Name'] if by_position['TE'] else 'None'}")

            # Find FLEX
            flex_candidates = []
            flex_candidates.extend(by_position['RB'][2:])
            flex_candidates.extend(by_position['WR'][3:])
            flex_candidates.extend(by_position['TE'][1:])

            if flex_candidates:
                best_flex = max(flex_candidates, key=lambda x: x['ETR Full PPR'])
                print(f"FLEX: {best_flex['Name']}")
            else:
                print("FLEX: None")

            print(f"DST: {by_position['DST'][0]['Name'] if by_position['DST'] else 'None'}")
            print(f"K: {by_position['K'][0]['Name'] if by_position['K'] else 'None'}")

        return sorted_results


def main():
    # Initialize simulator
    csv_path = r"C:\Users\musel\OneDrive\Desktop\Sports Stuff\Fantasy football\Temp_Rankings_Redraft_Auction 2025.csv"

    try:
        simulator = FantasyFootballDraftSimulator(csv_path)

        # Run all simulations
        results = simulator.run_all_simulations()

        # Display results
        final_rankings = simulator.display_results(results)

        print(f"\n{'=' * 60}")
        print("SUMMARY: BEST DRAFT POSITIONS FOR JARED")
        print(f"{'=' * 60}")

        for rank, (position, data) in enumerate(final_rankings[:5], 1):
            print(f"{rank}. Position {position}: {data['score']:.2f} points")

    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at the specified path.")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()