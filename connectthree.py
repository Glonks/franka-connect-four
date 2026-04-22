import numpy as np

class ConnectThree():
    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height
        # dictionary of board state -> (visit count, win count)
        self.nodes = np.load('connectthree.npy',allow_pickle='TRUE').item()
        self.board_state = "0"*(self.width * self.height)
        self.max_games = 10000

    def reset_game(self):
        self.board_state = "0"*(self.width * self.height)

    def user_turn(self):
        col = None
        # Store valid user input in col
        while col is None:
            input_str = input("Play which column?\n").split()[0]
            if input_str.isnumeric() and self.is_valid_move(int(input_str)-1):
                col = int(input_str)-1
            else:
                print("Error: invalid column number")
        
        # Place piece
        return self.play_move('2', col)
    
    def random_turn(self, piece):
        current_state = self.board_state
        for col in range(self.width):
            move = self.play_move(piece, col)
            self.board_state = move
            if self.check_victory() == piece:
                self.board_state = current_state
                return move
            self.board_state = current_state
        
        col = np.random.randint(self.width)
        while not self.is_valid_move(col):
            col = np.random.randint(self.width)
        return self.play_move(piece, col)
    
    def play_move(self, piece, col):
        # Place piece
        col_ix = self.board_ix(0, col)
        column = self.board_state[col_ix : col_ix + self.height]
        play_height = column.find('0')
        return self.board_state[:col_ix + play_height] + piece + self.board_state[col_ix + play_height + 1:]
    
    def mcdt(self):
        current_state = self.board_state
        
        for _ in range(self.max_games):
            history = [current_state]
            if current_state in self.nodes:
                visits, wins = self.nodes[current_state]
                self.nodes[current_state] = (visits + 1, wins)
            else:
                self.nodes[current_state] = (1, 0)

            win = '0'
            while win == '0':
                valid_moves = [self.play_move('1', col) for col in range(self.width) if self.is_valid_move(col)]

                # Check if every move has been explored before
                not_seen = [state for state in valid_moves if state not in self.nodes]

                move = None

                # If every child node has been explored, follow a heuristic
                if not_seen == []:
                    parent = history[-1]
                    parent_visits, _ = self.nodes[parent]
                    max_score = 0
                    max_state = 0
                    for state in valid_moves:
                        visits, wins = self.nodes[state]
                        score = (wins / visits) + np.sqrt(np.log(parent_visits) / visits)
                        
                        if score >= max_score:
                            max_score = score
                            max_state = state
                    
                    move = max_state
                # Otherwise explore a new node
                else:
                    move = str(np.random.choice(not_seen))
            
                # Make the decided-upon move
                history.append(move)
                if move in self.nodes:
                    visits, wins = self.nodes[move]
                    self.nodes[move] = (visits + 1, wins)
                else:
                    self.nodes[move] = (1, 0)
                self.board_state = move

                win = self.check_victory()
                if win == '1':
                    for state in history:
                        visits, wins = self.nodes[state]
                        self.nodes[state] = (visits, wins + 1)
                if win != '0':
                    break
                
                # Simulated opponent move
                opp_play = self.random_turn('2')
                self.board_state = opp_play
                win = self.check_victory()

            self.board_state = current_state

        # Select move with highest win rate
        valid_moves = [self.play_move('1', col) for col in range(self.width) if self.is_valid_move(col)]
        max_score = 0
        max_state = 0
        for state in valid_moves:
            visits, wins = self.nodes[state]
            score = (wins / visits)
            
            if score >= max_score:
                max_score = score
                max_state = state
        
        return max_state

        

    def full_turn(self):
        # Print the board
        state_str = ""
        for i in range(self.height):
            for j in range(self.width):
                state_str += self.board_state[self.board_ix(self.height - i - 1, j)]
            state_str += '\n'
        print(state_str)

        # Take user input
        self.board_state = self.user_turn()

        # Check for game end
        result = self.check_victory()
        if result == '2':
            self.reset_game()
            print("You win!")
            np.save('connectthree.npy', self.nodes)
            return 2
        if result == '-1':
            self.reset_game()
            print("It's a draw!")
            np.save('connectthree.npy', self.nodes)
            return -1
        
        # Play CPU turn
        self.board_state = self.mcdt()
        
        # Check for game end
        result = self.check_victory()
        if result == '1':
            self.reset_game()
            print("You lose!")
            np.save('connectthree.npy', self.nodes)
            return 1
        if result == '-1':
            self.reset_game()
            print("It's a draw!")
            np.save('connectthree.npy', self.nodes)
            return -1
        
        print('')
        return 0

    def board_ix(self, i, j):
        return self.height * j + i
    
    def is_valid_move(self, col_num):
        if col_num > self.width:
            return False
        
        return self.board_state[self.board_ix(self.height - 1, col_num)] == '0'
    
    def check_victory(self):
        for i in range(self.width):
            for j in range(self.height):
                piece = self.board_state[self.board_ix(i, j)]
                if piece == '0':
                    continue

                # Vertical
                if i > 0 and self.board_state[self.board_ix(i-1, j)] == piece \
                         and i < self.height - 1 and self.board_state[self.board_ix(i+1, j)] == piece:
                    return piece
                
                # Horizontal
                if j > 0 and self.board_state[self.board_ix(i, j-1)] == piece \
                         and j < self.width - 1 and self.board_state[self.board_ix(i, j+1)] == piece:
                    return piece
                
                # Diagonals
                if i > 0 and i < self.height - 1 and j > 0 and j < self.width - 1:
                    if self.board_state[self.board_ix(i-1, j-1)] == piece \
                       and self.board_state[self.board_ix(i+1, j+1)] == piece:
                        return piece
                    
                    if self.board_state[self.board_ix(i+1, j-1)] == piece \
                       and self.board_state[self.board_ix(i-1, j+1)] == piece:
                        return piece
        
        if self.board_state.find('0') == -1:
            return '-1'
        else:
            return '0'

def main():
    ct = ConnectThree()
    while ct.full_turn() == 0:
        pass


if __name__ == "__main__":
    main()