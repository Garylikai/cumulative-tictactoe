#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang (zhangshangtong.cpp@gmail.com)          #
# 2016 Jan Hakenberg (jan.hakenberg@gmail.com)                        #
# 2016 Tian Jun (tianjun.cpp@gmail.com)                               #
# 2016 Kenta Shimada (hyperkentakun@gmail.com)                        #
# 2022 - 2026 Kai Li (kai.li@stonybrook.edu)                          #
# 2026 Wei Zhu (wei.zhu@stonybrook.edu)                               #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# -----------------------------------------------------------------------
# cumulative_tictactoe_core.py
#
# Core game mechanics for cumulative tic-tac-toe:
#  - Board constants and TCD heuristic
#  - State representation and hashing
#  - STATE_CACHE (precomputed reachable states)
#  - Judger class
#  - Player (TD) and HumanPlayer classes
# -----------------------------------------------------------------------

import pickle
import numpy as np

# Board configuration constants
BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


def tcd(state_data):
    # Compute normalized triplet-coverage difference (TCD) for a given board.
    # Returns a value in [-1, 1].
    triplets_no_Os = 0
    triplets_no_Xs = 0
    # rows and columns
    for i in range(BOARD_ROWS):
        row = state_data[i, :]
        col = state_data[:, i]
        if np.all(row != -1):
            triplets_no_Os += 1
        if np.all(row != 1):
            triplets_no_Xs += 1
        if np.all(col != -1):
            triplets_no_Os += 1
        if np.all(col != 1):
            triplets_no_Xs += 1
    # main diagonal
    diag = np.diag(state_data)
    anti = np.diag(np.fliplr(state_data))
    if np.all(diag != -1):
        triplets_no_Os += 1
    if np.all(diag != 1):
        triplets_no_Xs += 1
    if np.all(anti != -1):
        triplets_no_Os += 1
    if np.all(anti != 1):
        triplets_no_Xs += 1
    # normalize using theoretical bounds [-2, 4]
    raw_diff = triplets_no_Os - triplets_no_Xs
    return (raw_diff - (-2)) / (4 - (-2)) * 2 - 1  # scale to [-1, 1]


class State:
    # Represents a board state in cumulative tic-tac-toe.
    def __init__(self):
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        self.winner = None
        self.hash_val = None
        self.end = None

    def hash_state(self):
        # Compute and cache a unique base-3 hash for the board state.
        if self.hash_val is None:
            h = 0
            for val in np.nditer(self.data):
                h = h * 3 + int(val) + 1
            self.hash_val = h
        return self.hash_val
  
    def is_terminal(self):
        # Determine if the play has ended and set winner and end flag.
        if self.end is not None:
            return self.end
        
        # Check if board is full
        if np.count_nonzero(self.data) != BOARD_SIZE:
            self.end = False
            return False
        
        # Evaluate scores for rows, columns, and diagonals
        results = []
        results.extend(np.sum(self.data, axis=1))  # rows
        results.extend(np.sum(self.data, axis=0))  # columns
        results.append(np.trace(self.data))  # diagonal
        results.append(np.trace(np.fliplr(self.data)))  # antidiagonal
        
        p1_score = sum(1 for i in results if i == BOARD_ROWS)
        p2_score = sum(1 for i in results if i == -BOARD_ROWS)
        
        if p1_score > p2_score:
            self.winner = 1
        elif p1_score < p2_score:
            self.winner = -1
        else:
            self.winner = 0

        self.end = True
        return True
        
    def next_state(self, i, j, symbol):
        # Return a new State after placing symbol at (i, j).
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    def print_state(self):
        # Print the board in human-readable format.
        sep = "-" * (BOARD_COLS * 4 + 1)
        for i in range(BOARD_ROWS):
            print(sep)
            row_symbols = []
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = "X"
                elif self.data[i, j] == -1:
                    token = "O"
                else:
                    token = " "
                row_symbols.append(f" {token} ")
            print("|" + "|".join(row_symbols) + "|")
        print(sep)


def get_STATE_CACHE():
    # Enumerate all reachable board states from the empty board.
    def recurse(state: State, symbol: int):
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    new_state = state.next_state(i, j, symbol)
                    h = new_state.hash_state()
                    if h not in STATE_CACHE:
                        is_terminal = new_state.is_terminal()
                        STATE_CACHE[h] = (new_state, is_terminal)
                        if not is_terminal:
                            recurse(new_state, -symbol)

    initial_symbol = 1
    initial_state = State()
    STATE_CACHE = dict()
    STATE_CACHE[initial_state.hash_state()] = (initial_state, initial_state.is_terminal())
    recurse(initial_state, initial_symbol)
    return STATE_CACHE


# Precompute all game states
STATE_CACHE = get_STATE_CACHE()


class Judger:
    # Orchestrates games between two players.
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    def reset(self):
        # Reset both players
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    def play(self, print_state=False, print_end_state=False):
        # Play a full game. Return winner (1, -1, or 0).
        # If print_state is True, print the state after each move.
        # If print_end_state is True, print the final state.
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        while True:
            player = next(alternator)
            i, j, symbol = player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash_state()
            current_state, is_terminal = STATE_CACHE[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()
            if is_terminal:
                if print_end_state:
                    current_state.print_state()
                return current_state.winner


class Player:
    # Tabular TD-learning agent with epsilon-greedy policy.
    def __init__(self, epsilon=0.05, step_size=0.50, heuristic="tcd"):
        self.value_table = dict()
        self.epsilon = epsilon
        self.step_size = step_size
        self.heuristic = heuristic
        self.states = []
        self.symbol = 0

    def reset(self):
        # Clear history of visited states.
        self.states = []
        self.was_greedy_move = []

    def set_state(self, state):
        # Record current state for backup and action selection.
        self.states.append(state)

    def set_symbol(self, symbol):
        # Initialize value estimates for all states from this player's perspective.
        self.symbol = symbol
        for hash_val in STATE_CACHE:
            state, is_terminal = STATE_CACHE[hash_val]
            if is_terminal:
                if state.winner == self.symbol:
                    self.value_table[hash_val] = 1
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.value_table[hash_val] = 0
                else:
                    self.value_table[hash_val] = -1
            elif self.heuristic == "tcd":
                self.value_table[hash_val] = tcd(state.data) * symbol
            elif self.heuristic == "random":
                # Random heuristic: assign a random value between -1 and 1
                self.value_table[hash_val] = np.random.uniform(-1, 1)
            elif self.heuristic == "zero":
                self.value_table[hash_val] = 0

    def backup(self):
        # Perform one-step TD backups over recorded states.
        states = [state.hash_state() for state in self.states]
        for i in reversed(range(len(states) - 1)):
            state = states[i]
            td_error = self.value_table[states[i+1]] - self.value_table[state]
            self.value_table[state] += self.step_size * td_error

    def act(self):
        # Choose an action via epsilon-was_greedy_move policy.
        state = self.states[-1]
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(i, j, self.symbol).hash_state())

        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            return action
        else:
            values = []
            for hash_val, pos in zip(next_states, next_positions):
                values.append((self.value_table[hash_val], pos))
            np.random.shuffle(values)
            values.sort(key=lambda x: x[0], reverse=True)
            action = values[0][1]
            action.append(self.symbol)
            return action

    def save_policy(self):
        # Persist value estimates to disk.
        with open("policy_%s.bin" % ("first" if self.symbol == 1 else "second"), "wb") as f:
            pickle.dump(self.value_table, f)

    def load_policy(self):
        # Load value estimates from disk.
        with open("policy_%s.bin" % ("first" if self.symbol == 1 else "second"), "rb") as f:
            self.value_table = pickle.load(f)


class HumanPlayer:
    # Human interface: map key inputs to board positions.
    # -------------
    # | q | w | e |
    # -------------
    # | a | s | d |
    # -------------
    # | z | x | c |
    # -------------
    def __init__(self):
        self.symbol = None
        self.keys = ["q", "w", "e", "a", "s", "d", "z", "x", "c"]
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol

    def act(self):
        # Prompt human for move and return (i, j, symbol).
        while True:
            try:
                # Print the board so the human can see current state
                self.state.print_state()
                key = input("Enter move (q/w/e a/s/d z/x/c) or 'quit' to exit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nInput aborted by user.")
                raise
        
            if key == "quit":
                print("User requested exit.")
                raise SystemExit
            
            # Validate key is recognized
            if key not in self.keys:
                print(f"Invalid key '{key}'. Valid keys: {', '.join(self.keys)}")
                continue

            # Map key to board coordinates
            data = self.keys.index(key)
            i = data // BOARD_COLS
            j = data % BOARD_COLS
            
            # Check that the chosen cell is empty
            if self.state.data[i, j] != 0:
                print(f"Cell ({i},{j}) is already occupied. Choose another key.")
                continue            
            
            # Valid move found; return coordinates with current symbol
            return i, j, self.symbol
