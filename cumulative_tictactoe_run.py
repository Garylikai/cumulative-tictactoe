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
# cumulative_tictactoe_run.py
#
# Training, competition and play entry points for cumulative tic-tac-toe.
# Imports core game implementation from cumulative_tictactoe_core.py.
# -----------------------------------------------------------------------

import numpy as np

# Import core classes and functions (must be in the same directory)
from cumulative_tictactoe_core import Player, HumanPlayer, Judger

# Training function
def train(episodes, print_every_n=500, window_size=20000, perf_threshold=1e-8, seed=None):
    # Train two TD agents via self-play.
    np.random.seed(seed)
    
    player1 = Player()
    player2 = Player()
    judger = Judger(player1, player2)
    
    # Counters for performance metrics
    p1_wins = 0
    p2_wins = 0
    draws = 0
    # Record running performance after each episode as (win_rate, draw_rate)
    performance_history = []
    episodes_to_converge = None

    for episode in range(1, episodes + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        else:
            draws += 1
        
        # Calculate running averages
        p1_win_rate = p1_wins / episode
        p2_win_rate = p2_wins / episode
        draw_rate = draws / episode
        performance_history.append((p1_win_rate, draw_rate))
        
        if episode % print_every_n == 0:
            print("Episode %d: P1 win: %.2f, P2 win: %.2f, Draw: %.2f" % (episode, p1_win_rate, p2_win_rate, draw_rate))
        
        player1.backup()
        player2.backup()
        judger.reset()

        # Check for convergence if we have enough episodes
        if episode >= window_size:
            # Consider the last window_size episodes
            recent_history = performance_history[-window_size:]
            # Separate win rates and draw rates
            win_rates_window = np.array([win[0] for win in recent_history])
            draw_rates_window = np.array([draw[1] for draw in recent_history])
            
            # Compute sample variance over the window. A low sample variance suggests stabilization.
            win_var = np.var(win_rates_window, ddof=1)
            draw_var = np.var(draw_rates_window, ddof=1)
            
            if win_var < perf_threshold and draw_var < perf_threshold:
                episodes_to_converge = episode
                print("Convergence achieved at episode %d with win rate sample variance %.4f and draw rate sample variance %.4f" % (episodes_to_converge, win_var, draw_var))
                break
        
    # Save the policy even if convergence was not reached within episodes
    player1.save_policy()
    player2.save_policy()

    if episodes_to_converge is None:
        print("Convergence not reached in %d episodes." % episodes)
    return episodes_to_converge

def compete(turns):
    # Evaluate trained policies over a number of matches (AI vs. AI).
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print("\n%d turns, player 1 win %.2f, player 2 win %.2f" % (turns, player1_win / turns, player2_win / turns))


def play(human_first=True):
    # Evaluate trained policies over a number of matches (Human vs. AI).
    while True:
        human = HumanPlayer()
        ai = Player(epsilon=0)
        if human_first:
            player1 = human
            player2 = ai
        else:
            player1 = ai
            player2 = human
        judger = Judger(player1, player2)
        ai.load_policy()
        winner = judger.play(print_end_state=True)
        if winner == human.symbol:
            print("You win!")
        elif winner == ai.symbol:
            print("You lose!")
        else:
            print("It is a tie!")


if __name__ == '__main__':
    # Training and Analysis
    train(int(1e6))
    compete(int(1e2))
    play()
