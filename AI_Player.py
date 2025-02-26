import numpy as np
import socket
import pickle
import time
from reversi import reversi

# cell weights
WEIGHTS = np.array([
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100],
])

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()
    # our_turn = None  # Will be set to 1 (white) or -1 (black)

    while True:
        # Receive play request from the server
        data = game_socket.recv(4096)
        if not data:
            break
        turn, board = pickle.loads(data)

        # Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return

        

        # our_turn = turn

        
        # Start timing
        start_time = time.time()
        time_limit = 5.0
        game.board = board.copy()
        best_move = (-1, -1)
        max_depth_reached = 0

        # Iterative deepening search
        for depth in range(1, 100):
            if time.time() - start_time > time_limit - 0.1:
                break
            result = alphabeta(game.board, depth, -float('inf'), float('inf'), True, turn, start_time, time_limit)
            if result is not None:
                value, move = result
                if move is not None:
                    best_move = move
                    max_depth_reached = depth
            else:
                break
        
        # TODO: Comment print to improve performance
        # print(f"Best move: {best_move}, Depth reached: {max_depth_reached}")
        # Send your move to the server. Send (-1, -1) to pass if no valid moves
        game_socket.send(pickle.dumps(list(best_move)))
        

def alphabeta(board, depth, alpha, beta, maximizingPlayer, player, start_time, time_limit):
    # Check time limit
    if time.time() - start_time > time_limit - 0.1:
        return None

    # Check if depth is zero
    if depth == 0:
        eval = evaluate_board(board, player)
        return eval, None

    valid_moves = get_valid_moves(board, player if maximizingPlayer else -player)
    if not valid_moves:
        eval = evaluate_board(board, player)
        return eval, None

    best_move = None
    if maximizingPlayer:
        value = -float('inf')
        for move in valid_moves:
            x, y = move
            new_board = board.copy()
            game = reversi()
            game.board = new_board
            game.step(x, y, player)
            result = alphabeta(game.board, depth-1, alpha, beta, False, player, start_time, time_limit)
            if result is None:
                return None
            eval, _ = result
            if eval > value:
                value = eval
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff
        return value, best_move
    else:
        value = float('inf')
        for move in valid_moves:
            x, y = move
            new_board = board.copy()
            game = reversi()
            game.board = new_board
            game.step(x, y, -player)
            result = alphabeta(game.board, depth-1, alpha, beta, True, player, start_time, time_limit)
            if result is None:
                return None
            eval, _ = result
            if eval < value:
                value = eval
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha cutoff
        return value, best_move

def get_valid_moves(board, player):
    game = reversi()
    game.board = board.copy()
    valid_moves = []
    for x in range(8):
        for y in range(8):
            if board[x, y] == 0:
                flip_count = game.step(x, y, player, commit=False)
                if flip_count >= 1:
                    # Heuristic for move ordering (prefer higher weights)
                    move_weight = WEIGHTS[x, y]
                    valid_moves.append((x, y, move_weight))
    # Sort moves based on move_weight for better pruning
    valid_moves.sort(key=lambda x: -x[2])
    # Return list of (x, y)
    return [(move[0], move[1]) for move in valid_moves]

def evaluate_board(board, player):
    opp_player = -player
    # piece difference score
    player_tiles = np.sum(board == player)
    opp_tiles = np.sum(board == opp_player)
    piece_difference_score = 100 * (player_tiles - opp_tiles) / (player_tiles + opp_tiles + 1)

    # Mobility (number of legal moves)
    player_moves = len(get_valid_moves(board, player))
    opp_moves = len(get_valid_moves(board, opp_player))
    if player_moves + opp_moves != 0:
        mobility = 100 * (player_moves - opp_moves) / (player_moves + opp_moves + 1)
    else:
        mobility = 0

    # Corner occupancy
    corners = [(0,0), (0,7), (7,0), (7,7)]
    player_corners = sum([1 for x, y in corners if board[x, y] == player])
    opp_corners = sum([1 for x, y in corners if board[x, y] == opp_player])
    corner_score = 25 * (player_corners - opp_corners)

    # Positional weights
    positional = np.sum(board * WEIGHTS * player)

    total = piece_difference_score + mobility + corner_score + positional
    return total

if __name__ == '__main__':
    main()
