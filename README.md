# Reversi-AI-NonDeepLearning

Construct an AI for Reversi using only non-deep learning methods(use alpha betha pruning) that interact with the provided environment (reversi.py, reversi_server.py, greedy_player.py), with the goal of consistently defeating the greedy player, which relies on greedy algorithm techniques.

## Rules:

- Standard Reversi rules (Check https://en.wikipedia.org/wiki/ReversiLinks to an external site.)
- Except white goes first instead of black.
- Code must be written in Python
- No multi-threading
- No machine learning
- No more than 5 seconds per hand.

## To Use:

Please simply run the code with no arguments, just like the greedy_player.py

## Methodology:

In this project, we implemented an AI player for Reversi using **alpha-beta pruning** as our non-deep-learning approach. Below is a brief overview of the methodology:

- ### Iterative Deepening Search  
  To balance computation time and search depth, we use iterative deepening. The algorithm begins with a shallow search (depth 1) and incrementally increases the depth (up to 100) until the 5-second time limit nears (with a 0.1-second buffer). This ensures the AI always has a valid move by retaining the best move from the deepest completed search, even if time runs short.

- ### Alpha-Beta Pruning  
  The core decision-making uses the alpha-beta pruning algorithm, a variant of minimax that optimizes by reducing the number of evaluated nodes in the game tree. It tracks two bounds—`alpha` (best value for the maximizing player) and `beta` (best value for the minimizing player)—and skips branches where further exploration won’t change the result. This is essential for staying within the time constraint.

- ### Heuristic Evaluation Function  
  At the search depth limit or a terminal state, the board is scored using a weighted combination of four factors:  
  - **Piece Difference**: Relative count of player vs. opponent tiles, normalized as a percentage (weight: 100).  
  - **Mobility**: Difference in legal moves between player and opponent, favoring control (weight: 100).  
  - **Corner Occupancy**: Value of controlling corners (weight: 25 per corner), due to their stability and strategic importance.  
  - **Positional Weights**: A predefined `WEIGHTS` matrix assigns values to tiles—corners (100), edges (5-10), and risky spots near corners (-20, -50)—scaled by the player’s color.  

  The total score sums these factors, steering the AI toward strong positions.

- ### Move Ordering  
  Valid moves are sorted by their positional weights from the `WEIGHTS` matrix to boost pruning efficiency. High-value moves (e.g., corners) are evaluated first, increasing the chances of early cutoffs in the alpha-beta search.

- ### Time Management  
  The algorithm tracks elapsed time with `time.time()` and halts deeper iterations if the 5-second limit approaches. This guarantees timely responses to the server.

- ### Game Logic  
  The `reversi` module manages board updates and move validation. During evaluation, the AI simulates moves on a copied board to preserve the original state.
