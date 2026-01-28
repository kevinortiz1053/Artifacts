#SUDOKO SOLVER
board = {
    "A": [6, 0, 8, 7, 0, 2, 1, 0, 0],
    "B": [4, 0, 0, 0, 1, 0, 0, 0, 2],
    "C": [0, 2, 5, 4, 0, 0, 0, 0, 0],
    "D": [7, 0, 1, 0, 8, 0, 4, 0, 5],
    "E": [0, 8, 0, 0, 0, 0, 0, 7, 0],
    "F": [5, 0, 9, 0, 6, 0, 3, 0, 1],
    "G": [0, 0, 0, 0, 0, 6, 7, 5, 0],
    "H": [2, 0, 0, 0, 9, 0, 0, 0, 8],
    "I": [0, 0, 6, 8, 0, 5, 2, 0, 3]
}


def get_domains(board):  # getting domains of each individual small box based on a starting board
    domains = {}
    do = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    for k in do:
        for i in range(9):
            if board[k][i] == 0:
                n = k + str(i+1)
                domains[n] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                n = k + str(i+1)
                domains[n] = [board[k][i]]
    return domains


# x = get_domains(board)
# print(x)
# print(len(x))


def gen_all_arcs():
    arcs = []
    do = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    # arcs for Alldiff(A1,A2...,A9) Alldiff(B1,B2...,B9) etc.
    for j in range(9):
        l = do[j]
        for i in range(9):
            n = l + str(i+1)
            for k in range(i+1, 9):
                m = l + str(k+1)
                arcs.append([n, m])
                arcs.append([m, n])
    # #arcs for Alldiff(A1,B1...,I1) Alldiff(A2,B2...,I2) etc.
    for j in range(9):
        for i in range(9):
            n = do[j] + str(i+1)
            for k in range(j+1, 9):
                m = do[k] + str(i+1)
                arcs.append([n, m])
                arcs.append([m, n])
    #arcs for Alldiff(A1,A2,A3...C1,C2,C3) etc. for all the 9-box units
    for a in range(0, 9, 3):
        for k in range(0, 9, 3):
            temp = []
            for i in range(k, k+3):
                for j in range(a, a+3):
                    temp.append(do[i] + str(j+1))
            for i in range(9):
                n = temp[i]
                for j in range(i+1, 9):
                    m = temp[j]
                    arcs.append([n, m])
                    arcs.append([m, n])

    return arcs  # this is my initial queue


# x = gen_all_arcs()
# print(x)
# print(len(x))


def revise(arc, domains):
    revised = False
    d1 = domains[arc[0]]
    d2 = domains[arc[1]]
    for i in d1:
        for j in d2:
            # only time constraint would NOT hold is if domain 2 only has 1 value and domain 1 has that value
            if i == j and len(d2) == 1:
                ind = d1.index(i)
                d1.pop(ind)
                domains[arc[0]] = d1
                revised == True
    return revised, domains


# def AC3(board):
#     domains = get_domains(board)
#     queue = gen_all_arcs()
#     while len(queue) != 0:
#         arc = queue.pop(0)
#         r, domains = revise(arc, domains)
#         if r == True:
#             if len(arc[0]) == 0:
#                 return False, board
#             for i in queue:
#                 if i[1] == arc[0] and i[0] != arc[1]:
#                     queue.append(i)
#     do = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
#     for k in range(9):  # building the board after making everything arc consistent
#         for n in range(9):
#             board[do[k]][n] = domains[do[k] + str(n+1)]
#
#     return True, board

def AC3(board):
    #domains = get_domains(board)
    queue = gen_all_arcs()
    while len(queue) != 0:
        arc = queue.pop(0)
        r, domains = revise(arc, board)
        if r == True:
            if len(arc[0]) == 0:
                return False
            for i in queue:
                if i[1] == arc[0] and i[0] != arc[1]:
                    queue.append(i)
    # do = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    # for k in range(9):  # building the board after making everything arc consistent
    #     for n in range(9):
    #         board[do[k]][n] = domains[do[k] + str(n+1)]

    return True


# bb, bbb = AC3(board)


# x, y = AC3(board)
# print(y)

def select_var(board):
    y = 10
    for i in board:
        var = board[i]
        if len(var) != 1:
            x = len(var)
            if x < y:
                final_var = i
                y = x
    return final_var


def assignment_check(var, v, board):
    all_arcs = gen_all_arcs()  # doesn't respect loose coupling, flaw in design
    list_ind = int(var[1]) - 1
    for j in all_arcs:
        if j[0] == var:
            list_ind1 = int(j[1][1]) - 1
            for k in board[j[1][0]][list_ind1]:
                if v == k and len(board[j[1][0]][list_ind1]) == 1:
                    print(var + ":" + str(v) + " is in "
                          + str(j[1]) + ":" + str(board[j[1][0]][list_ind1]))
                    return False
    return True


def check_full(cboard):
    count = 0
    for i in cboard:
        if len(cboard[i]) != 1:
            count = count + 1

    if count == 0:  # returning assignment if all variables only have one possible value
        print("got one")
        print(cboard)
        return True
    return False


def back_track(cboard):
    board_copy = cboard.copy()
    if check_full(cboard):
        return True, cboard
    print(cboard)
    var = select_var(cboard)
    values = cboard[var]
    for v in values:
        print("this is board_copy for:" + var)
        board_copy[var] = [v]
        print(board_copy)
        #if assignment_check(var, v, board_copy):
        if AC3(board_copy):
            result, result_board = back_track(board_copy)
            if result:
                return result, result_board
    return False, cboard


def sudoku(board):
    dboard = get_domains(board)
    result, result_board = back_track(dboard)
    if not result:
        print("Couldn't find solution")
        return False, board
    print("Found Solution")
    print(result_board)
    return True, result_board


sudoku(board)
