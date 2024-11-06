# Modifying Author Team: Drew York and Mario V. Valenta
# CPSC6420: Artificial Intelligence
# Homework Assignment #3
# Due Date: 11/6/2022
# python version 3.9
#
# ORIGINAL CODE INFORMATION:
# connect4.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)
#

"""
In this assignment, the task is to implement the minimax algorithm with depth
limit for a Connect-4 game.

To complete the assignment, you must finish these functions:
    minimax (line 196), alphabeta (line 237), and expectimax (line 280)
in this file.

In the Connect-4 game, two players place discs in a 6-by-7 board one by one.
The discs will fall straight down and occupy the lowest available space of
the chosen column. The player wins if four of his or her discs are connected
in a line horizontally, vertically or diagonally.
See https://en.wikipedia.org/wiki/Connect_Four for more details about the game.

A Board() class is provided to simulate the game board.
It has the following properties:
    b.rows          # number of rows of the game board
    b.cols          # number of columns of the game board
    b.PLAYER1       # an integer flag to represent the player 1
    b.PLAYER2       # an integer flag to represent the player 2
    b.EMPTY_SLOT    # an integer flag to represent an empty slot in the board;

and the following methods:
    b.terminal()            # check if the game is terminal
                            # terminal means draw or someone wins

    b.has_draw()            # check if the game is a draw

    w = b.who_wins()        # return the winner of the game or None if there
                            # is no winner yet 
                            # w should be in [b.PLAYER1,b.PLAYER2, None]

    b.occupied(row, col)    # check if the slot at the specific location is
                            # occupied

    x = b.get(row, col)     # get the player occupying the given slot
                            # x should be in [b.PLAYER1, b.PLAYER2, b.EMPTY_SLOT]

    row = b.row(r)          # get the specific row of the game described using
                            # b.PLAYER1, b.PLAYER2 and b.EMPTY_SLOT

    col = b.column(r)       # get a specific column of the game board

    b.placeable(col)        # check if a disc can be placed at the specific
                            # column

    b.place(player, col)    # place a disc at the specific column for player
        # raise ValueError if the specific column does not have available space
    
    new_board = b.clone()   # return a new board instance having the same
                            # disc placement with b

    str = b.dump()          # a string to describe the game board using
                            # b.PLAYER1, b.PLAYER2 and b.EMPTY_SLOT
Hints:
    1. Depth-limited Search
        We use depth-limited search in the current code. That is we
    stop the search forcefully, and perform evaluation directly not only when a
    terminal state is reached but also when the search reaches the specified
    depth.
    2. Game State
        Three elements decide the game state. The current board state, the
    player that needs to take an action (place a disc), and the current search
    depth (remaining depth).
    3. Evaluation Target
        The minimax algorithm always considers that the adversary tries to
    minimize the score of the max player, for whom the algorithm is called
    initially. The adversary never considers its own score at all during this
    process. Therefore, when evaluating nodes, the target should always be
    the max player.
    4. Search Result
        The pesudo code provided in the slides only returns the best utility value.
    However, in practice, we need to select the action that is associated with this
    value. Here, such action is specified as the column in which a disc should be
    placed for the max player. Therefore, for each search algorithm, you should
    consider all valid actions for the max player, and return the one that leads 
    to the best value. 

"""

# use math library if needed
import math

def get_child_boards(player, board):
    """
    Generate a list of succesor boards obtained by placing a disc 
    at the given board for a given player
   
    Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the player that will place a disc on the board
    board: the current board instance

    Returns
    -------
    a list of (col, new_board) tuples,
    where col is the column in which a new disc is placed (left column has a 0 index), 
    and new_board is the resulting board instance
    """
    res = []
    for c in range(board.cols):
        if board.placeable(c):
            tmp_board = board.clone()
            tmp_board.place(player, c)
            res.append((c, tmp_board))
    return res


def evaluate(player, board):
    """
    This is a function to evaluate the advantage of the specific player at the
    given game board.

    Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the specific player
    board: the board instance

    Returns
    -------
    score: float
        a scalar to evaluate the advantage of the specific player at the given
        game board
    """
    adversary = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
    # Initialize the value of scores
    # [s0, s1, s2, s3, --s4--]
    # s0 for the case where all slots are empty in a 4-slot segment
    # s1 for the case where the player occupies one slot in a 4-slot line, the rest are empty
    # s2 for two slots occupied
    # s3 for three
    # s4 for four
    score = [0]*5
    adv_score = [0]*5

    # Initialize the weights
    # [w0, w1, w2, w3, --w4--]
    # w0 for s0, w1 for s1, w2 for s2, w3 for s3
    # w4 for s4
    weights = [0, 1, 4, 16, 1000]

    # Obtain all 4-slot segments on the board
    seg = []
    invalid_slot = -1
    left_revolved = [
        [invalid_slot]*r + board.row(r) + \
        [invalid_slot]*(board.rows-1-r) for r in range(board.rows)
    ]
    right_revolved = [
        [invalid_slot]*(board.rows-1-r) + board.row(r) + \
        [invalid_slot]*r for r in range(board.rows)
    ]
    for r in range(board.rows):
        # row
        row = board.row(r) 
        for c in range(board.cols-3):
            seg.append(row[c:c+4])
    for c in range(board.cols):
        # col
        col = board.col(c) 
        for r in range(board.rows-3):
            seg.append(col[r:r+4])
    for c in zip(*left_revolved):
        # slash
        for r in range(board.rows-3):
            seg.append(c[r:r+4])
    for c in zip(*right_revolved): 
        # backslash
        for r in range(board.rows-3):
            seg.append(c[r:r+4])
    # compute score
    for s in seg:
        if invalid_slot in s:
            continue
        if adversary not in s:
            score[s.count(player)] += 1
        if player not in s:
            adv_score[s.count(adversary)] += 1
    reward = sum([s*w for s, w in zip(score, weights)])
    penalty = sum([s*w for s, w in zip(adv_score, weights)])
    return reward - penalty


def minimax(player, board, depth_limit):
    """
    Minimax algorithm with limited search depth.

    Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the player that needs to take an action (place a disc in the game)
    board: the current game board instance
    depth_limit: int
        the tree depth that the search algorithm needs to go further before stopping
    max_player: boolean

    Returns
    -------
    placement: int or None
        the column in which a disc should be placed for the specific player
        (counted from the most left as 0)
        None to give up the game
    """
    max_player = player
    placement = None

### Please finish the code below ##############################################
###############################################################################
    def value(player, board, depth_limit):
        nonlocal placement
        print("Called value with player=" + str(player) + " and depth=" + str(depth_limit))
        depth_limit -= 1        

        if player != max_player:
            val = min_value(player, board, depth_limit)
        else:
            val = max_value(player, board, depth_limit)
        placement = val[1]
        return val[0]

    def max_value(player, board, depth_limit):
        placement = None
        print("Running max_value player=" + str(player) + " at depth=" + str(depth_limit))
        if depth_limit == 0:
            val = evaluate(max_player, board)
            print("Evaluated player=" + str(player) + " MaxVal=" + str(val) + " With this board:\n" + str(board))
        else:  
            val = -math.inf
            successors = get_child_boards(player, board)

            for child in successors:
                print("For child =" + str(child) + " with next_player=" + str(player))
                v_child = value(next_player, child[1], depth_limit)
                print("Max_value v_child=" + str(v_child))
                print("Max_Value val=" + str(val))
                if v_child > val:
                    val = v_child
                    print("New Max_val? " + str(val))
                    placement = child[0]
                    print("Placement? " + str(placement)) 
        print("Returning MaxVal=" + str(val) + " and placement=" + str(placement) + " at depth=" + str(depth_limit))
        return val, placement
    
    def min_value(player, board, depth_limit):
        placement = None
        print("Running min_value player=" + str(player) + " at depth=" + str(depth_limit))
        if depth_limit == 0:
            val = evaluate(max_player, board)
            print("Evaluated player=" + str(player) + " MinVal=" + str(val) + " With this board:\n" + str(board))
        else:
            val = math.inf
            successors = get_child_boards(player, board)

            for child in successors:
                print("For child =" + str(child) + " with next_player=" + str(player))
                v_child = value(max_player, child[1], depth_limit)
                print("Min_value v_child=" + str(v_child))
                print("Min_Value val=" + str(val))
                if v_child < val:
                    val = v_child
                    placement = child[0]
                    print("New Min_val? " + str(val))
                    print("Placement? " + str(placement))
        print("Returning MinVal=" + str(val) + " at depth=" + str(depth_limit))
        return val, placement

    next_player = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
    score = -math.inf

    val = value(player, board, (depth_limit +1))
    print("Final MaxVal=" + str(val) + " and placement=" + str(placement))

###############################################################################
    return placement


def alphabeta(player, board, depth_limit):
    """
    Minimax algorithm with alpha-beta pruning.

     Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the player that needs to take an action (place a disc in the game)
    board: the current game board instance
    depth_limit: int
        the tree depth that the search algorithm needs to go further before stopping
    alpha: float
    beta: float
    max_player: boolean


    Returns
    -------
    placement: int or None
        the column in which a disc should be placed for the specific player
        (counted from the most left as 0)
        None to give up the game
    """
    max_player = player
    placement = None

### Please finish the code below ##############################################
###############################################################################
    alpha = -math.inf
    beta = math.inf
    def value(player, board, depth_limit, alpha, beta):
        nonlocal placement
        print("Called value with player=" + str(player) + " and depth=" + str(depth_limit))
        depth_limit -= 1

        if player != max_player:
            val = min_value(player, board, depth_limit, alpha, beta)
        else:
            val = max_value(player, board, depth_limit, alpha, beta)
        placement = val[1]
        return val[0]

    def max_value(player, board, depth_limit, alpha, beta):
        placement = None
        print("Running max_value player=" + str(player) + " at depth=" + str(depth_limit))
        if depth_limit == 0:
            val = evaluate(max_player, board)
            print("Evaluated player=" + str(player) + " MaxVal=" + str(val) + " With this board:\n" + str(board))
        else:
            val = -math.inf
            successors = get_child_boards(player, board)

            for child in successors:
                print("For child =" + str(child) + " with next_player=" + str(player))
                v_child = value(next_player, child[1], depth_limit, alpha, beta)
                print("Max_value v_child=" + str(v_child))
                print("Max_Value val=" + str(val))
                if v_child > val:
                    val = v_child
                    placement = child[0]
                    print("New Max_val? " + str(val))
                if val >= beta:
                    print("Pruning!! MaxVal =" + str(val) + " Beta =" + str(beta) + " placement=" + str(placement) + " at_depth=" + str(depth_limit))
                    break
                if val > alpha:
                    alpha = val
                    print("Alpha is now: " + str(alpha))

        print("Returning MaxVal=" + str(val) + " and placement=" + str(placement) + " at depth=" + str(depth_limit))
        return val, placement

    def min_value(player, board, depth_limit, alpha, beta):
        placement = None
        print("Running min_value player=" + str(player) + " at depth=" + str(depth_limit))
        if depth_limit == 0:
            val = evaluate(max_player, board)
            print("Evaluated player=" + str(player) + " MinVal=" + str(val) + " With this board:\n" + str(board))
        else:
            val = math.inf
            successors = get_child_boards(player, board)

            for child in successors:
                print("For child =" + str(child) + " with next_player=" + str(player))
                v_child = value(max_player, child[1], depth_limit, alpha, beta)
                print("Min_value v_child=" + str(v_child))
                print("Min_Value val=" + str(val))
                if v_child < val:
                    val = v_child
                    placement = child[0]
                    print("New Min_val? " + str(val))
                    print("Placement? " + str(placement))
                if val <= alpha:
                    print("Pruning!! MinVal =" + str(val) + " Alpha =" + str(alpha) + " at depth=" + str(depth_limit))
                    break
                if val < beta:
                    beta = val
                    print("Beta is now: " + str(beta))
        print("Returning MinVal=" + str(val) + " at depth=" + str(depth_limit))
        return val, placement

    next_player = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
    score = -math.inf

    val = value(player, board, (depth_limit +1), alpha, beta)
    print("Final MaxVal=" + str(val) + " and placement=" + str(placement))

###############################################################################
    return placement





def expectimax(player, board, depth_limit):
    """
    Expectimax algorithm.
    We assume that the adversary of the initial player chooses actions
    uniformly at random.
    Say that it is the turn for Player 1 when the function is called initially,
    then, during search, Player 2 is assumed to pick actions uniformly at
    random.

    Parameters
    ----------
    player: board.PLAYER1 or board.PLAYER2
        the player that needs to take an action (place a disc in the game)
    board: the current game board instance
    depth_limit: int
        the tree depth that the search algorithm needs to go before stopping
    max_player: boolean

    Returns
    -------
    placement: int or None
        the column in which a disc should be placed for the specific player
        (counted from the most left as 0)
        None to give up the game
    """
    max_player = player
    placement = None

### Please finish the code below ##############################################
###############################################################################
    def value(player, board, depth_limit):
        nonlocal placement
        print("Called value with player=" + str(player) + " and depth=" + str(depth_limit))
        depth_limit -= 1

        if player != max_player:
            val = min_value(player, board, depth_limit)
        else:
            val = max_value(player, board, depth_limit)
        placement = val[1]
        return val[0]

    def max_value(player, board, depth_limit):
        placement = None
        print("Running max_value player=" + str(player) + " at depth=" + str(depth_limit))
        if depth_limit == 0:
            val = evaluate(max_player, board)
            print("Evaluated player=" + str(player) + " MaxVal=" + str(val) + " With this board:\n" + str(board))
        else: 
            val = -math.inf
            successors = get_child_boards(player, board)

            for child in successors:
                print("For child =" + str(child) + " with next_player=" + str(player))
                v_child = value(next_player, child[1], depth_limit)
                print("Max_value v_child=" + str(v_child))
                print("Max_Value val=" + str(val))
                if v_child > val:
                    val = v_child
                    print("New Max_val? " + str(val))
                    placement = child[0]
                    print("Placement? " + str(placement))
        print("Returning MaxVal=" + str(val) + " and placement=" + str(placement) + " at depth=" + str(depth_limit))
        return val, placement

    def min_value(player, board, depth_limit):
        print("Running min_value player=" + str(player) + " at depth=" + str(depth_limit))
        if depth_limit == 0:
            val = evaluate(max_player, board)
            print("Evaluated player=" + str(player) + " MinVal=" + str(val) + " With this board:\n" + str(board))
        else:
            val = math.inf
            val_sum = 0
            successors = get_child_boards(player, board)

            for child in successors:
                print("For child =" + str(child) + " with next_player=" + str(player))
                v_child = value(max_player, child[1], depth_limit)
                print("Min_value v_child=" + str(v_child))
                print("Min_Value val=" + str(val))
                val_sum = val_sum + v_child

            if len(successors) != 0: 
                val = val_sum / len(successors)
            else:
                val = evaluate(max_player, board) 
        return val, None

    next_player = board.PLAYER2 if player == board.PLAYER1 else board.PLAYER1
    score = -math.inf

    val = value(player, board, (depth_limit +1))
    print("Final val=" + str(val))
###############################################################################
    return placement


if __name__ == "__main__":
    from utils.app import App
    import tkinter

    algs = {
        "Minimax": minimax,
        "Alpha-beta pruning": alphabeta,
        "Expectimax": expectimax
    }

    root = tkinter.Tk()
    App(algs, root)
    root.mainloop()
