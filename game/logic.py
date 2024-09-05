# from https://github.com/rajitbanerjee/2048-pygame/blob/master/logic.py

def shiftLeft(board):
    # remove 0's in between numbers
    for i in range(4):
        nums, count = [], 0
        for j in range(4):
            if board[i][j] != 0:
                nums.append(board[i][j])
                count += 1
        board[i] = nums + [0 for _ in range(4-count)]
        # board[i].extend([0] * (4 - count))

def shiftRight(board):
    # remove 0's in between numbers
    for i in range(4):
        nums, count = [], 0
        for j in range(4):
            if board[i][j] != 0:
                nums.append(board[i][j])
                count += 1
        board[i] = [0] * (4 - count)
        board[i].extend(nums)

def rotateLeft(board):
    b = [[board[j][i] for j in range(4)] for i in range(3, -1, -1)]
    return b

def rotateRight(board):
    b = rotateLeft(board)
    b = rotateLeft(b)
    return rotateLeft(b)