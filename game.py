import pygame
import sys
from collections import defaultdict
from itertools import chain
import numpy as np

class Game: 
    display_width = 750
    display_height = 550
    grid_width = 11
    grid_height = 11
    cell_width = 50
    board_state = []
    show_display = True

    player_turn = 0
    start_new_turn = False

    def initialize_game(players=[]): 
        Game.display_screen = pygame.display.set_mode((Game.display_width, Game.display_height))
        Game.value_font = pygame.font.SysFont(None, 20)
        Game.text_font = pygame.font.SysFont(None, 40)
        Game.text_font2 = pygame.font.SysFont(None, 30)

        Game.rounds = 0
        Game.winner = None
        Game.players = players
        Game.player_turn = 0 
        Game.active_player = Game.get_active_player()

        Game.board_state = Game.get_board_state(Game.players[0].pieces, Game.players[1].pieces)
        Game.legal_actions = defaultdict(list) 

        Game.active_player.choose_action(Game.get_current_game_state())

    def game_loop():
        game_state = Game.get_current_game_state()
        Game.test_game_over(game_state)

        if Game.start_new_turn: 
            Player.start_turn(game_state)
            Game.active_player.num_pieces = game_state["num_pieces"]
            Game.start_new_turn = False

        if not Player.turn_over(game_state): 
            Game.active_player.choose_action(game_state)
            Game.active_player.num_pieces = game_state["num_pieces"]
        else: 
            Game.player_turn += 1
            if Game.player_turn == len(Game.players): 
                Game.player_turn = 0
                Game.rounds += 1 
            Game.active_player = Game.get_active_player()
            Game.start_new_turn = True
           
        if Game.show_display:
            Game.display_game(Game.get_board_state(Game.players[0].pieces, Game.players[1].pieces))
        
    def test_game_over(game_state): 
        if Player.no_actions(game_state["snake_dict"], game_state["num_pieces"]) or game_state["spawn_pos"] in game_state["op_piece_dict"]: 
            print(f"Game Over, Winner: {Game.get_inactive_player().player_name}")  
            Game.winner = Game.get_inactive_player()
        if Player.no_actions(game_state["op_snake_dict"], game_state["op_num_pieces"]) or game_state["op_spawn_pos"] in game_state["piece_dict"]: 
            print(f"Game Over, Winner: {Game.active_player.player_name}")  
            Game.winner = Game.active_player

    def get_board_state(p1_pieces, p2_pieces): 
        grid = [0]*(Game.grid_width*Game.grid_height)

        for piece, data in p1_pieces.items(): 
            grid[Game.pos_to_grid_index(piece)] = data[0]
        for piece, data in p2_pieces.items(): 
            grid[Game.pos_to_grid_index(piece)] = -data[0]
        return grid
    
    def get_current_game_state(): 
        return {"spawn_pos": Game.active_player.spawn_pos, "num_pieces": Game.active_player.num_pieces, "snake_dict": Game.active_player.snakes, "piece_dict": Game.active_player.pieces, 
                "op_spawn_pos": Game.get_inactive_player().spawn_pos, "op_num_pieces": Game.get_inactive_player().num_pieces, "op_snake_dict": Game.get_inactive_player().snakes, "op_piece_dict": Game.get_inactive_player().pieces}

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
        return Game.players[(Game.player_turn+1)%2 ]

    def get_player(index):
        return Game.players[index]
    
    def cell_empty(game_state, pos): 
        return pos not in game_state["piece_dict"] and pos not in game_state["op_piece_dict"]
    
    def print_board_state(board_state): 
        grid = [[0] * Game.grid_width for _ in range(Game.grid_height)]
        for i, num in enumerate(board_state):
            row = i % Game.grid_height
            col = i // Game.grid_height
            grid[row][col] = num
        for row in grid:
            print(" ".join(f"{num:2}" for num in row))

    def display_game(game_state):   
        board_state = Game.get_board_state(game_state)

        Game.display_screen.fill((255, 255, 255))
        for player in Game.players:
            player.display()
 
        for indx, val in enumerate(board_state): 
            pos = Game.grid_index_to_pos(indx)
            color = (100, 100, 100)
            if val > 0: 
                color = Game.players[0].color
                if game_state["piece_dict"][Game.grid_index_to_pos(indx)][1]: 
                    
            if val < 0: 
                color = Game.players[1].color
            
            pygame.draw.rect(Game.display_screen, color, (pos[0]*Game.cell_width+1, pos[1]*Game.cell_width+1, 50, 50), width=1)
            if val != 0: 
                value_text_surface = Game.value_font.render(f"{abs(val)}", False, (0,0,0))
                Game.display_screen.blit(value_text_surface, (pos[0]*Game.cell_width+21, pos[1]*Game.cell_width+18))
        
        Actions.display()

        text_surface = Game.text_font.render(f"Turn: {Game.active_player.player_name}", True, (0,0,0))
        Game.display_screen.blit(text_surface, (570, 50))
        text_surface2 = Game.text_font2.render(f"Pieces: {Game.active_player.num_pieces}", True, (0,0,0))
        Game.display_screen.blit(text_surface2, (570, 100))
        
        pygame.display.flip()
    
class Player: 
    
    def __init__(self, id, spawn_pos, num_pieces, color, name): 
        self.id = id
        self.spawn_pos = spawn_pos
        self.num_pieces = num_pieces
        self.color = color
        self.player_name = name
        self.active = True
    
        #{snake_id: [piece_coord1, piece_coord2, piece_coord...]}
        #{1: [(0,1),(2,2),(3,4)], 2: [(5,5),(9,7)], ...}
        self.snakes = defaultdict(list)

        #{(piece_coords): [value, is_active, snake_id]}
        #pieces = {(0,1): [6,True,1], (2,2): [6,True,1], (3,4): [6,True,1], (5,5): [6,True,2], (9,7): [6,True,2], ...}
        self.pieces = defaultdict(list)

    def choose_action(self, game_state): 
        ...
        
    def run_action(action_code, game_state, legal_actions): 
        if action_code in chain.from_iterable(legal_actions.values()): 
            action, positions = Actions.translate_code(action_code)
            action(positions, game_state)

    def get_actions(game_state):
        actions = defaultdict(list) 

        #get placements
        placement_actions = []
        if game_state["num_pieces"] > 0: 
            placement_actions = [Actions.get_placement_code(game_state["spawn_pos"])] if game_state["spawn_pos"] not in game_state["piece_dict"] else Snake.get_placement_actions(game_state["snake_dict"][game_state["piece_dict"][game_state["spawn_pos"]][2]], game_state) 
        actions["p"].extend(placement_actions)
        
        #get movements and rolls
        for snake_pieces in game_state["snake_dict"].values():
            snake_actions = Snake.get_actions(snake_pieces, game_state)
            actions["m"].extend(snake_actions["m"])
            actions["r"].extend(snake_actions["r"])

        return actions    

    #Actions
    def roll_piece(positions, game_state): 
        value = np.random.randint(1, 7) 
        game_state["piece_dict"][positions[0]][0] = value
        game_state["piece_dict"][positions[0]][1] = True
        Snake.check_matching_values(Snake.get_pieces_from_pos(positions[0], game_state["snake_dict"], game_state["piece_dict"]), game_state, game_state["piece_dict"], num_pieces="num_pieces")

    def move_piece(positions, game_state): 
        to_move, move_loc = positions
        to_move_val = game_state["piece_dict"][to_move][0]
        if move_loc in game_state["op_piece_dict"]: 
            Snake.remove_piece_from_dict(move_loc, game_state["op_snake_dict"], game_state["op_piece_dict"])
        Snake.add_piece_to_dict(move_loc, to_move_val, game_state["snake_dict"], game_state["piece_dict"])
        Snake.remove_piece_from_dict(to_move, game_state["snake_dict"], game_state["piece_dict"])
        Snake.make_pieces_inactive(game_state["snake_dict"][game_state["piece_dict"][move_loc][2]], game_state["piece_dict"])

    def place_piece(positions, game_state): 
        Snake.add_piece_to_dict(positions[0], 1, game_state["snake_dict"], game_state["piece_dict"])
        game_state["num_pieces"] -= 1

    def start_turn(game_state): 
        Player.activate_pieces(game_state["piece_dict"])
        Player.check_all_matching_snakes(game_state, game_state["snake_dict"], game_state["piece_dict"], num_pieces="num_pieces")

    def check_all_matching_snakes(game_state, snake_dict, piece_dict, num_pieces="num_pieces"): 
        for snake_pieces in snake_dict.values(): 
            Snake.check_matching_values(snake_pieces, game_state, piece_dict, num_pieces="num_pieces")

    def activate_pieces(piece_dict): 
        for piece in piece_dict: 
            piece_dict[piece][1] = False 

    def remove_all_pieces(piece_dict): 
        piece_dict.clear() 

    def total_pieces(piece_dict): 
        return len(piece_dict) 
    
    def turn_over(game_state): 
        an_active_snake = any([Snake.is_active(snake_pieces, game_state["piece_dict"]) for snake_pieces in game_state["snake_dict"].values()])
        return len(game_state["snake_dict"]) > 0 and not an_active_snake
    
    def no_actions(snake_dict, num_pieces): 
        longest_snake = max([len(snake_pieces) for snake_pieces in snake_dict.values()]) if snake_dict else 0
        return num_pieces == 0 and longest_snake <= 1 
    
    def display(self): 
        pygame.draw.circle(Game.display_screen, self.color, (self.spawn_pos[0]*50+25, self.spawn_pos[1]*50+25), 20, width=1)

class Snake: 
    snake_id = 0 

    def get_actions(snake_pieces, game_state): 
        actions = defaultdict(list)
        if not Snake.is_active(snake_pieces, game_state["piece_dict"]): 
            return actions
        actions["r"].extend(Snake.get_roll_actions(snake_pieces, game_state))
        actions["m"].extend(Snake.get_movement_actions(snake_pieces, game_state))
        return actions
    
    def get_movement_actions(snake_pieces, game_state): 
        movements = [] 
        if Snake.num_inactive(snake_pieces, game_state["piece_dict"]) != 1: 
            return movements

        for piece_to_move in snake_pieces: 
            if len(Piece.get_connections(piece_to_move, snake_pieces)) == 1: 
                perimeter = list(set(chain.from_iterable([Piece.get_non_connections(piece, snake_pieces) for piece in snake_pieces if piece != piece_to_move])))
                legal_move_positions = [pos for pos in perimeter if pos not in game_state["op_piece_dict"] or game_state["piece_dict"][piece_to_move][0] >= game_state["op_piece_dict"][pos][0]]
                movements.extend(Actions.get_movement_codes(piece_to_move, legal_move_positions))
        return movements  

    def get_roll_actions(snake_pieces, game_state): 
        rolls = []
        for piece_to_roll in snake_pieces: 
            if not game_state["piece_dict"][piece_to_roll][1]: 
                rolls.append(Actions.get_roll_code(piece_to_roll))
        return rolls

    def get_placement_actions(snake_pieces, enemy_pieces_dict): 
        return Actions.get_placement_codes(Snake.get_empty_perimeter(snake_pieces, enemy_pieces_dict))

    def add_piece_to_dict(pos, val, snake_dict, piece_dict): 
        connected_snakes = Piece.get_connected_snakes(pos, piece_dict) 

        #create parent snake 
        snake_id = Snake.get_unique_id() if not connected_snakes else Snake.get_combined_snakes(connected_snakes, snake_dict, piece_dict) 

        #add piece to the snake and piece dictionaries
        snake_dict[snake_id].append(pos)
        piece_dict[pos] = [val, False, snake_id]

    def remove_piece_from_dict(pos, snake_dict, piece_dict): 
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
                Snake.snake_search(pos=connections[0], visited=snake_pieces, to_search=snake_dict[snake_id_to_remove], to_find=connections, snake_id=snake_id, piece_dict=piece_dict)
                snake_dict[snake_id] = snake_pieces
            snake_dict.pop(snake_id_to_remove)
        elif Snake.is_empty(snake_id_to_remove, snake_dict): 
            snake_dict.pop(snake_id_to_remove)

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
    
    def check_matching_values(pieces, game_state, piece_dict, num_pieces="num_pieces"): 
        if piece_dict[pieces[0]][0] == 1 or len(pieces) == 1: 
            return 
        all_matching = all([piece_dict[piece][0] == piece_dict[pieces[0]][0] for piece in pieces])
        if all_matching: 
            for piece in pieces: 
                piece_dict[piece][0] = 1 
                piece_dict[piece][1] = True
            game_state[num_pieces] += 1

    def is_active(pieces, piece_dict): 
        if len(pieces) == 1: 
            return not piece_dict[pieces[0]][1]
        else:
            return sum([piece_dict[piece][1] for piece in pieces]) < 2 
    
    def num_inactive(pieces, piece_dict):
        return sum([piece_dict[piece][1] for piece in pieces])
    
    def make_pieces_inactive(pieces, piece_dict): 
        for piece in pieces: 
            piece_dict[piece][1] = True

    def get_pieces_from_pos(pos, snake_dict, piece_dict): 
        return snake_dict[piece_dict[pos][2]]

    def is_empty(snake_id, snake_dict): 
        return len(snake_dict[snake_id]) == 0 

class Piece:
    rel_edge_positions = [(1,0), (-1,0), (0,1), (0,-1)]
    
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
        state_indx = Game.pos_to_grid_index(selected_cell)
        game_state = Game.get_current_game_state()
        legal_actions = Player.get_actions(game_state)

        if event.button == 1: 
            if Game.cell_empty(game_state, selected_cell): 
                Player.run_action(Actions.get_placement_code(selected_cell), game_state, legal_actions)
            else: 
                Player.run_action(Actions.get_roll_code(selected_cell), game_state, legal_actions)
                
        if event.button == 3: 
            if Actions.store_input: 
                Player.run_action(Actions.get_movement_code(Actions.store_input, selected_cell), game_state, legal_actions)
                Actions.store_input = None 
            elif selected_cell in Game.active_player.pieces: 
                Actions.store_input = selected_cell

        Game.active_player.num_pieces = game_state["num_pieces"]

    def space_action(event): 
        Game.show_display = not Game.show_display 

    def translate_code(action_code): 
        action = action_code.split("-")
        action_type = action[0]
        if action_type == "p": 
            return Player.place_piece, [(int(action[1]), int(action[2]))]
        if action_type == "m":
            return Player.move_piece, [(int(action[1]), int(action[2])), (int(action[3]), int(action[4]))]
        if action_type == "r":  
            return Player.roll_piece, [(int(action[1]), int(action[2]))]
    
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

    def display():
        mouse_pos = pygame.mouse.get_pos()
        if Actions.store_input: 
            ...      
    
                    
        
        
        
        


        
        
        
        