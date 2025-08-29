import pygame
import sys
from collections import defaultdict
from itertools import chain
import copy 
import numpy as np

class Game: 
    display_width = 750
    display_height = 550
    grid_width = 11
    grid_height = 11
    cell_width = 50
    show_display = True

    def initialize_game(players, initial_state): 
        Game.display_screen = pygame.display.set_mode((Game.display_width, Game.display_height))
        Game.value_font = pygame.font.SysFont(None, 20)
        Game.text_font = pygame.font.SysFont(None, 40)
        Game.text_font2 = pygame.font.SysFont(None, 30)

        Game.game_state = GameState(initial_state) 
        Game.rounds = 0
        Game.winner = None
        Game.players = players
        Game.player_turn = 0 
        Game.active_player = Game.get_active_player()

        Game.active_player.choose_action(Game.game_state)

    def game_loop():
        game_state = Game.game_state

        winner = game_state.get_winner()
        if winner is not None: 
            Game.winner = Game.players[winner]
            print(f"Game Over, Winner: {Game.players[winner].player_name}")

        if not game_state.turn_over(Game.player_turn): 
            Game.active_player.choose_action(game_state)
        else: 
            Game.player_turn += 1
            if Game.player_turn == len(Game.players): 
                Game.player_turn = 0
                Game.rounds += 1 
            Game.active_player = Game.get_active_player()
            game_state.start_turn(Game.player_turn)
           
        if Game.show_display:
            Game.display_game()
    
    def get_game_state(): 
        return Game.game_state

    def valid_search_pos(pos): 
        return pos[0] >= 0 and pos[0] <= Game.grid_width-1 and pos[1] >= 0 and pos[1] <= Game.grid_height-1

    def grid_search(pos, grid): 
        return grid[Game.pos_to_grid_index(pos)]
    
    def pos_to_grid_index(pos): 
        return pos[0]*Game.grid_height+pos[1]
    
    def grid_index_to_pos(indx): 
        return (indx//Game.grid_height, indx%Game.grid_height)
    
    def coords_to_grid_pos(coords): 
        return (coords[0]//Game.cell_width, coords[1]//Game.cell_width)

    def get_active_player(): 
        return Game.players[Game.player_turn]
    
    def get_inactive_player(): 
        return Game.players[(Game.player_turn+1)%2]
    
    def get_other_player(player):
        return Game.players[(player+1)%2]

    def get_player(index):
        return Game.players[index]
    
    def print_board_state(board_state): 
        grid = [[0] * Game.grid_width for _ in range(Game.grid_height)]
        for i, num in enumerate(board_state):
            row = i % Game.grid_height
            col = i // Game.grid_height
            grid[row][col] = num
        for row in grid:
            print(" ".join(f"{num:2}" for num in row))

    def display_game():   
        entire_state = Game.game_state.get_data()
        board_state = Game.game_state.get_board_state()
        players = Game.players

        Game.display_screen.fill((255, 255, 255))

        for indx, val in enumerate(board_state): 
            pos = Game.grid_index_to_pos(indx)
            color = (100, 100, 100)
            player = Game.game_state.get_player_from_pos(pos)

            if player is not None: 
                if entire_state[player]["piece_dict"][pos][1]: 
                    pygame.draw.rect(Game.display_screen, Game.players[player].get_inactive_color(), (pos[0]*Game.cell_width+1, pos[1]*Game.cell_width+1, 50-1, 50-1))
                color = Game.players[player].color
                value_text_surface = Game.value_font.render(f"{abs(val)}", False, (0,0,0))
                Game.display_screen.blit(value_text_surface, (pos[0]*Game.cell_width+21, pos[1]*Game.cell_width+18))

            pygame.draw.rect(Game.display_screen, color, (pos[0]*Game.cell_width+1, pos[1]*Game.cell_width+1, 50, 50), width=1)

        for player in entire_state: 
            pygame.draw.circle(Game.display_screen, players[player].color, (entire_state[player]["spawn_pos"][0]*50+25, entire_state[player]["spawn_pos"][1]*50+25), 20, width=1)
                
        Actions.display()

        text_surface = Game.text_font.render(f"Turn: {Game.active_player.player_name}", True, (0,0,0))
        Game.display_screen.blit(text_surface, (570, 50))
        text_surface2 = Game.text_font2.render(f"Pieces: {Game.game_state.get_player_data(Game.player_turn)["num_pieces"]}", True, (0,0,0))
        Game.display_screen.blit(text_surface2, (570, 100))
        pygame.display.flip()

class GameState:
    placements_per_turn = 3

    def __init__(self, entire_state): 
        self.entire_state = entire_state

    def __getitem__(self, key): 
        if key in self.entire_state: 
            return self.entire_state[key]
        else: 
            raise KeyError(f"Key '{key}' not found.")
    
    #Get Actions
    def get_actions(self, player):
        actions = defaultdict(list) 
        player_state = self.entire_state[player]

        #get placements
        placement_actions = []
        if player_state["num_pieces"] > 0 and player_state["num_placements"] > 0: 
            placement_actions = [Actions.get_placement_code(player_state["spawn_pos"])] if player_state["spawn_pos"] not in player_state["piece_dict"] else self.get_snake_placements(player_state["piece_dict"][player_state["spawn_pos"]][2], player)
        actions["p"].extend(placement_actions)

        #get movements and rolls
        for snake in player_state["snake_dict"]:
            if Snake.is_active(player_state["snake_dict"][snake], player_state["piece_dict"]):
                actions["m"].extend(self.get_snake_movements(snake, player))
                actions["r"].extend(self.get_snake_rolls(snake, player))

        return actions

    def get_snake_movements(self, snake, player): 
        snake_pieces, piece_dict, enemy_pieces = self.entire_state[player]["snake_dict"][snake], self.entire_state[player]["piece_dict"], self.entire_state[(player+1)%2]["piece_dict"]
        movements = [] 
        if Snake.num_inactive(snake_pieces, piece_dict) != 1: 
            return movements

        for piece_to_move in snake_pieces: 
            if len(Piece.get_connections(piece_to_move, snake_pieces)) == 1: 
                perimeter = list(set(chain.from_iterable([Piece.get_non_connections(piece, snake_pieces) for piece in snake_pieces if piece != piece_to_move])))
                legal_move_positions = [pos for pos in perimeter if pos not in enemy_pieces or piece_dict[piece_to_move][0] >= enemy_pieces[pos][0]]
                movements.extend(Actions.get_movement_codes(piece_to_move, legal_move_positions))
        return movements  

    def get_snake_rolls(self, snake, player): 
        snake_pieces, piece_dict = self.entire_state[player]["snake_dict"][snake], self.entire_state[player]["piece_dict"]
        return Actions.get_roll_codes([piece for piece in snake_pieces if not piece_dict[piece][1]])

    def get_snake_placements(self, snake, player): 
        snake_pieces, enemy_pieces = self.entire_state[player]["snake_dict"][snake], self.entire_state[(player+1)%2]["piece_dict"]
        return Actions.get_placement_codes(Snake.get_empty_perimeter(snake_pieces, enemy_pieces))

    #Make Actions
    def run_action(self, player, action): 
        action_decode = action.split("-")
        type = action_decode[0]
        if type == "p": 
            self.place_piece((int(action_decode[1]), int(action_decode[2])), player)
        if type == "m":
            self.move_piece((int(action_decode[1]), int(action_decode[2])), (int(action_decode[3]), int(action_decode[4])), player)
        if type == "r":  
            self.roll_piece((int(action_decode[1]), int(action_decode[2])), None, player)
        if type == "dr": 
            self.roll_piece((int(action_decode[1]), int(action_decode[2])), int(action_decode[3]), player) 

    def roll_piece(self, pos, value, player): 
        piece_dict = self.entire_state[player]["piece_dict"]
        if not value: 
            value = np.random.randint(1, 7) 
        piece_dict[pos][0] = value
        piece_dict[pos][1] = True
        self.check_matching_values(piece_dict[pos][2], player)

    def move_piece(self, to_move, move_loc, player): 
        piece_dict, enemy_pieces = self.entire_state[player]["piece_dict"], self.entire_state[(player+1)%2]["piece_dict"]
        to_move_val = piece_dict[to_move][0]
        if move_loc in enemy_pieces: 
            self.remove_piece(move_loc, (player+1)%2)
        self.add_piece(move_loc, to_move_val, player)
        self.remove_piece(to_move, player)
        self.make_snake_inactive(piece_dict[move_loc][2], player)

    def place_piece(self, position, player): 
        player_state = self.entire_state[player]
        self.add_piece(position, 1, player)
        player_state["num_pieces"] -= 1
        player_state["num_placements"] -= 1

    def start_turn(self, player):
        self.activate_pieces(player)
        self.check_all_matching_snakes(player)
        self.entire_state[player]["num_placements"] = GameState.placements_per_turn

    def check_all_matching_snakes(self, player): 
        snake_dict = self.entire_state[player]["snake_dict"]
        for snake in snake_dict: 
            self.check_matching_values(snake, player)

    def activate_pieces(self, player): 
        piece_dict = self.entire_state[player]["piece_dict"]
        for piece in piece_dict: 
            piece_dict[piece][1] = False 

    def check_matching_values(self, snake, player): 
        player_state, snake_pieces, piece_dict = self.entire_state[player], self.entire_state[player]["snake_dict"][snake], self.entire_state[player]["piece_dict"] 
        if piece_dict[snake_pieces[0]][0] == 1 or len(snake_pieces) == 1: 
            return 
        all_matching = all([piece_dict[piece][0] == piece_dict[snake_pieces[0]][0] for piece in snake_pieces])
        if all_matching: 
            for piece in snake_pieces: 
                piece_dict[piece][0] = 1 
                piece_dict[piece][1] = True
            player_state["num_pieces"] += 1

    def make_snake_inactive(self, snake, player): 
        snake_pieces, piece_dict = self.entire_state[player]["snake_dict"][snake], self.entire_state[player]["piece_dict"]
        for piece in snake_pieces: 
            piece_dict[piece][1] = True

    def add_piece(self, pos, val, player): 
        snake_dict, piece_dict = self.entire_state[player]["snake_dict"], self.entire_state[player]["piece_dict"]
        connected_snakes = Piece.get_connected_snakes(pos, piece_dict) 

        #create parent snake 
        snake_id = Snake.get_unique_id() if not connected_snakes else Snake.get_combined_snakes(connected_snakes, snake_dict, piece_dict) 

        #add piece to the snake and piece dictionaries
        snake_dict[snake_id].append(pos)
        piece_dict[pos] = [val, False, snake_id]

    def remove_piece(self, pos, player): 
        snake_dict, piece_dict = self.entire_state[player]["snake_dict"], self.entire_state[player]["piece_dict"]
        if pos not in piece_dict: 
            return 
        connections = Piece.get_connections(pos, piece_dict)
        snake_id_to_remove = piece_dict[pos][2] 

        #remove piece 
        snake_dict[snake_id_to_remove].remove(pos)
        piece_dict.pop(pos)

        #handle split snakes 
        if len(connections) >= 2:
            while connections: 
                snake_id = Snake.get_unique_id()
                snake_pieces = []
                Snake.snake_search(connections[0], visited=snake_pieces, to_search=snake_dict[snake_id_to_remove], to_find=connections, snake_id=snake_id, piece_dict=piece_dict)
                snake_dict[snake_id] = snake_pieces
            snake_dict.pop(snake_id_to_remove)
        elif Snake.is_empty(snake_id_to_remove, snake_dict): 
            snake_dict.pop(snake_id_to_remove)

    #Get game features
    def get_winner(self): 
        for player in self.entire_state: 
            if self.no_actions(player) or self.spawn_occupied(player): 
                return (player+1)%2
        return None 

    def spawn_occupied(self, player):
        spawn_pos, enemy_pieces = self.entire_state[player]["spawn_pos"], self.entire_state[(player+1)%2]["piece_dict"]
        return spawn_pos in enemy_pieces

    def turn_over(self, player): 
        snake_dict, piece_dict = self.entire_state[player]["snake_dict"], self.entire_state[player]["piece_dict"]
        an_active_snake = any([Snake.is_active(snake_pieces, piece_dict) for snake_pieces in snake_dict.values()])
        return len(snake_dict) > 0 and not an_active_snake
    
    def no_actions(self, player): 
        snake_dict, num_pieces = self.entire_state[player]["snake_dict"], self.entire_state[player]["num_pieces"]
        longest_snake = max([len(snake_pieces) for snake_pieces in snake_dict.values()]) if snake_dict else 0
        return num_pieces == 0 and longest_snake <= 1 
    
    def get_total_pieces(self, player): 
        return len(self.entire_state[player]["piece_dict"])+self.entire_state[player]["num_pieces"]
    
    def get_num_immobile_snakes(self, player): 
        num_immobile = 0
        for snake_pieces in self.entire_state[player]["snake_dict"].values():
            if Snake.is_immobile(snake_pieces, self.entire_state[player]["piece_dict"]): 
                num_immobile += 1
        return num_immobile
    
    #Create States 
    def get_board_piece_state(self): 
        state = []
        state.extend(self.get_board_state())
        state.extend([self.entire_state[0]["num_pieces"], self.entire_state[1]["num_pieces"]])
        return state
    
    def get_board_state(self): 
        p1_pieces, p2_pieces = self.entire_state[0]["piece_dict"], self.entire_state[1]["piece_dict"]
        grid = [0]*(Game.grid_width*Game.grid_height)
        for piece, data in p1_pieces.items(): 
            grid[Game.pos_to_grid_index(piece)] = data[0]
        for piece, data in p2_pieces.items(): 
            grid[Game.pos_to_grid_index(piece)] = -data[0]
        return grid

    def generate_successor(self, player, action):
        new_state = GameState(self.get_data_copy())
        new_state.run_action(player, action)
        return new_state
    
    def flip_perspective(self): 
        x_line = (Game.grid_height//2) 
        y_line = (Game.grid_width//2)

        #symmetrically flip positions 
        for player in self.entire_state: 
            for snake in self.entire_state[player]["snake_dict"]: 
                new_pieces = [((2*x_line)-piece[0], (2*y_line)-piece[1]) for piece in self.entire_state[player]["snake_dict"][snake]]
                self.entire_state[player]["snake_dict"][snake] = new_pieces
            new_piece_dict = defaultdict(list)
            for piece in self.entire_state[player]["piece_dict"]: 
                new_piece = ((2*x_line)-piece[0], (2*y_line)-piece[1]) 
                new_piece_dict[new_piece] = self.entire_state[player]["piece_dict"][piece]
            self.entire_state[player]["piece_dict"] = new_piece_dict

        #switch player piece data 
        temp_piece_data = (self.entire_state[0]["snake_dict"], self.entire_state[0]["piece_dict"], self.entire_state[0]["num_pieces"])
        self.entire_state[0]["snake_dict"], self.entire_state[0]["piece_dict"], self.entire_state[0]["num_pieces"]  = self.entire_state[1]["snake_dict"], self.entire_state[1]["piece_dict"], self.entire_state[1]["num_pieces"]
        self.entire_state[1]["snake_dict"], self.entire_state[1]["piece_dict"], self.entire_state[1]["num_pieces"] = temp_piece_data
        
    #Helpful Methods 
    def get_data(self):
        return self.entire_state
    
    def get_copy(self):
        return GameState(self.get_data_copy())
    
    def get_data_copy(self): 
        return copy.deepcopy(self.entire_state)

    def get_player_data(self, player):
        return self.entire_state[player]
    
    def get_player_from_pos(self, pos): 
        for player in self.entire_state:
            if pos in self.entire_state[player]["piece_dict"]:
                return player
        return None
    
    def pos_empty(self, pos): 
        return all([pos not in self.entire_state[player]["piece_dict"] for player in list(self.entire_state)])
 
class Player: 
    
    def __init__(self, id, color, name): 
        self.id = id
        self.color = color
        self.player_name = name
        self.episode = []

    def choose_action(self, game_state): 
        ...
    
    def get_inactive_color(self):
        return (min([self.color[0]+200, 255]), min([self.color[1]+200, 255]), min([self.color[2]+200, 255]))
    
    def display(self, spawn_pos): 
        pygame.draw.circle(Game.display_screen, self.color, (spawn_pos[0]*50+25, spawn_pos[1]*50+25), 20, width=1)

class Snake: 
    snake_id = 0 

    def get_combined_snakes(ids_to_connect, snake_dict, piece_dict):
        if not ids_to_connect: 
            return None 
        if len(ids_to_connect) == 1:
            return ids_to_connect[0]
        
        # add pieces from all combined snakes to the first snake, 
        # reassign snake_ids of pieces, and remove every snake except the first 
        for id in ids_to_connect[1:]: 
            for piece in snake_dict[id]: 
                snake_dict[ids_to_connect[0]].append(piece)
                piece_dict[piece][2] = ids_to_connect[0]
            snake_dict.pop(id)
        
        return ids_to_connect[0] 
    
    def snake_search(pos, visited, to_search, to_find, snake_id, piece_dict):
        visited.append(pos)
        if pos in to_find:
            to_find.remove(pos)
        piece_dict[pos][2] = snake_id
        for piece in Piece.get_connections(pos, to_search):
            if piece not in visited: 
                Snake.snake_search(piece, visited, to_search, to_find, snake_id, piece_dict)

    def get_unique_id():
        Snake.snake_id += 1
        return Snake.snake_id 
    
    def get_perimeter(snake_pieces): 
        return list(set(chain.from_iterable([Piece.get_non_connections(piece, snake_pieces) for piece in snake_pieces]))) 

    def get_empty_perimeter(snake_pieces, enemy_pieces): 
        return list(set(chain.from_iterable([Piece.get_empty_adjacent(piece, snake_pieces, enemy_pieces) for piece in snake_pieces]))) 

    def is_immobile(pieces, piece_dict): 
        if len(pieces) < 4: 
            return False
        for piece in pieces: 
            if Piece.is_immobile(piece, piece_dict): 
                return True
        return False

    def is_active(pieces, piece_dict): 
        if len(pieces) == 1: 
            return not piece_dict[pieces[0]][1]
        else:
            return sum([piece_dict[piece][1] for piece in pieces]) < 2 
    
    def num_inactive(pieces, piece_dict):
        return sum([piece_dict[piece][1] for piece in pieces])

    def get_pieces_from_pos(pos, snake_dict, piece_dict): 
        return snake_dict[piece_dict[pos][2]]

    def is_empty(snake_id, snake_dict): 
        return len(snake_dict[snake_id]) == 0 

class Piece:
    rel_edge_positions = [(1,0), (-1,0), (0,1), (0,-1)]
    rel_corner_positions = [[(1,0), (0,1), (1,1)], [(1,0), (0,-1), (-1,-1)], [(-1,0), (0,1), (-1,1)], [(-1,0), (0,-1), (-1,-1)]]

    def is_immobile(pos, pieces): 
        for corner in Piece.rel_corner_positions: 
            if all([(pos[0]+rel_pos[0], pos[1]+rel_pos[1]) in pieces for rel_pos in corner]):
                return True 
        return False

    def get_connected_snakes(pos, pieces): 
        return list(set([pieces[(pos[0]+rel_edge[0], pos[1]+rel_edge[1])][2] for rel_edge in Piece.rel_edge_positions if (pos[0]+rel_edge[0], pos[1]+rel_edge[1]) in pieces]))

    def get_connections(pos, pieces): 
        return [(pos[0]+rel_edge[0], pos[1]+rel_edge[1]) for rel_edge in Piece.rel_edge_positions if (pos[0]+rel_edge[0], pos[1]+rel_edge[1]) in pieces]
        
    def get_non_connections(pos, pieces):
        return [(pos[0]+rel_edge[0], pos[1]+rel_edge[1]) for rel_edge in Piece.rel_edge_positions if Game.valid_search_pos((pos[0]+rel_edge[0], pos[1]+rel_edge[1])) and (pos[0]+rel_edge[0], pos[1]+rel_edge[1]) not in pieces]

    def get_empty_adjacent(pos, pieces1, pieces2): 
        return [(pos[0]+rel_edge[0], pos[1]+rel_edge[1]) for rel_edge in Piece.rel_edge_positions if Game.valid_search_pos((pos[0]+rel_edge[0], pos[1]+rel_edge[1])) and (pos[0]+rel_edge[0], pos[1]+rel_edge[1]) not in pieces1 and (pos[0]+rel_edge[0], pos[1]+rel_edge[1]) not in pieces2]

    def get_adjacent(pos): 
        return [(pos[0]+rel_edge[0], pos[1]+rel_edge[1]) for rel_edge in Piece.rel_edge_positions if Game.valid_search_pos((pos[0]+rel_edge[0], pos[1]+rel_edge[1]))]

class Actions:

    store_input = None

    def click_action(event):
        selected_cell = Game.coords_to_grid_pos(event.pos)
        game_state = Game.game_state
        player_turn = Game.player_turn
        legal_actions = list(chain.from_iterable(game_state.get_actions(player_turn).values()))

        if event.button == 1: 
            if game_state.pos_empty(selected_cell): 
                action = Actions.get_placement_code(selected_cell)
                if action in legal_actions: 
                    game_state.run_action(player_turn, action)
                    Game.active_player.episode.append(game_state.get_copy())
            else: 
                action = Actions.get_roll_code(selected_cell)
                if action in legal_actions: 
                    game_state.run_action(player_turn, action)
                    Game.active_player.episode.append(game_state.get_copy())
                
        if event.button == 3: 
            if Actions.store_input: 
                action = Actions.get_movement_code(Actions.store_input, selected_cell)
                if action in legal_actions: 
                    game_state.run_action(player_turn, action)
                    Game.active_player.episode.append(game_state.get_copy())
                Actions.store_input = None 
            elif selected_cell in game_state.get_player_data(player_turn)["piece_dict"]: 
                Actions.store_input = selected_cell

    def space_action(event): 
        Game.show_display = not Game.show_display   
        #Game.game_state.flip_perspective()
    
    def get_movement_code(pos1, pos2): 
        return "m"+"-"+str(pos1[0])+"-"+str(pos1[1])+"-"+str(pos2[0])+"-"+str(pos2[1])

    def get_movement_codes(to_move, locations):
        return [Actions.get_movement_code(to_move, loc) for loc in locations]
    
    def get_placement_code(pos): 
        return "p"+"-"+str(pos[0])+"-"+str(pos[1])

    def get_placement_codes(cells): 
        return [Actions.get_placement_code(cell) for cell in cells]

    def get_roll_code(pos): 
        return "r"+"-"+str(pos[0])+"-"+str(pos[1])

    def get_roll_codes(cells):
        return [Actions.get_roll_code(cell) for cell in cells]
    
    def get_droll_code(pos, val): 
        return "dr"+"-"+str(pos[0])+"-"+str(pos[1])+"-"+str(val)
    
    def get_droll_codes(roll_code): 
        action_decode = roll_code.split("-")
        type = action_decode[0]
        pos = (int(action_decode[1]), int(action_decode[2]))

        return [Actions.get_droll_code(pos, val) for val in range(1,7)]

    def display():
        mouse_pos = pygame.mouse.get_pos()

        if Actions.store_input:  
            pygame.draw.rect(Game.display_screen, Game.active_player.color, (mouse_pos[0]-10, mouse_pos[1]-10, 20, 20), width=1)
                  
    
                    
        
        
        
        


        
        
        
        