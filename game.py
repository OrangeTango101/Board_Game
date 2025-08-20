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
    game_state = []
    show_display = True

    player_turn = 0

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


        Game.game_state = Game.get_game_state(Game.players[0].pieces, Game.players[1].pieces)
        Game.legal_actions = defaultdict(list) 
        Game.legal_actions["p"] = ["p-5-10"] 

        Game.active_player.choose_action()

    def game_loop():
        if not Game.test_turn_over(): 
            Game.active_player.choose_action()
        Game.test_game_over()

        if Game.show_display:
            Game.display_game()

    def test_turn_over():
        #end turn if no legal actions 
        if not Game.legal_actions: 
            Game.next_turn()
            return True
        #continue turn if legal actions, but no pieces on board 
        if not Game.active_player.pieces: 
            return False
        #end turn if all snakes are inactive 
        if not any([Snake.is_active(pieces, Game.active_player) for pieces in Game.active_player.snakes.values()]): 
            Game.next_turn()
            return True 
        #continue turn if legal actions and active snakes 
        return False 
        
    def test_game_over(): 
        if sum([player.active for player in Game.players]) == 1: 
            print(f"Game Over, Winner: {Game.active_player.player_name}")  
            Game.winner = Game.active_player 

    def next_turn(): 
        Game.player_turn += 1
        
        if Game.player_turn == len(Game.players): 
            Game.player_turn = 0
            Game.rounds += 1 

        Game.active_player = Game.get_active_player()

        Game.active_player.activate_pieces()
        Game.legal_actions = Game.active_player.get_actions()
        
        #check if active 
        if Game.active_player.spawn_pos in Game.get_inactive_player().pieces or not Game.legal_actions: 
            Game.active_player.active = False
        
        #move to next player and remove current player if inactive 
        if not Game.active_player.active: 
            Game.active_player.delete_snakes()
            Game.next_turn()

    def update_turn(): 
        Game.legal_actions = Game.active_player.get_actions()
        Game.game_state = Game.get_game_state(Game.players[0].pieces, Game.players[1].pieces)

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
    
    def cell_empty(game_state, indx): 
        return game_state[indx] == 0
    
    def get_game_state(p1_pieces, p2_pieces): 
        grid = [0]*(Game.grid_width*Game.grid_height)

        for piece, data in p1_pieces.items(): 
            grid[Game.pos_to_grid_index(piece)] = data[0]
        for piece, data in p2_pieces.items(): 
            grid[Game.pos_to_grid_index(piece)] = -data[0]
        return grid

    def display_game():   
        Game.display_screen.fill((255, 255, 255))
        for player in Game.players:
            player.display()
 
        for indx, val in enumerate(Game.game_state): 
            pos = Game.grid_index_to_pos(indx)
            color = (100, 100, 100)
            if val > 0: 
                color = Game.players[0].color
            if val < 0: 
                color = Game.players[1].color

            pygame.draw.rect(Game.display_screen, color, (pos[0]*Game.cell_width+1, pos[1]*Game.cell_width+1, 50, 50), width=1)
            if val != 0: 
                value_text_surface = Game.value_font.render(f"{val}", False, (0,0,0))
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
        self.pieces = defaultdict(Piece.default_value)

    def choose_action(self): 
        ...
        
    def run_action(action_code, snake_dict, piece_dict): 
        if action_code in Game.legal_actions.values(): 
            action, data = Actions.translate_code(action_code)
            action(data, snake_dict, piece_dict)
            Game.update_turn()

    def get_actions(self):
        actions = defaultdict(list) 
        
        if not self.active:
            return []
        
        if self.spawn_pos not in self.pieces: 
            actions["p"].append(Actions.translate_placement(self.spawn_pos))

        #get placements
        for snake in self.snakes.values():
            snake_actions = Snake.get_actions(snake)
            actions["p"].extend(snake_actions["p"])
            actions["m"].extend(snake_actions["m"])
            actions["r"].extend(snake_actions["r"])
            
        return actions    

    #Actions
    def roll_piece(snake_dict, piece_dict, positions): 
        value = np.random.randint(1, 7) 
        ... 

    def move_piece(snake_dict, piece_dict, positions): 
        ...

    def place_piece(snake_dict, piece_dict, positions): 
        ...

    #Snake Methods
    def remove_snake(self, snake):
        ... 

    #Piece methods 
    def activate_pieces(self): 
        ... 
    def delete_all_pieces(self): 
        ... 
    def total_pieces(self): 
        ... 
    
    
    def display(self): 
        pygame.draw.circle(Game.display_screen, self.color, (self.spawn_pos[0]*50+25, self.spawn_pos[1]*50+25), 20, width=1)


class Snake: 
    snake_id = 0 

    def get_actions(snake_dict, piece_dict, enemy_pieces_dict): 
        actions = defaultdict(list)
        if not Snake.is_active(piece_dict): 
            return actions
        
    def get_legal_movements(self):
        actions = []
        if not self.active or self.rolls != 1: 
            return []
            
        for tail in self.get_tails(): 
            perimeter = list(set(chain.from_iterable([cell.perimeter for cell in self.cells if cell != tail]))) 
            legal_moves = [cell for cell in perimeter if cell.value <= tail.value]
            actions.extend(Actions.translate_movement_batch(tail, legal_moves))   
            
        return actions
        
    def get_roll_actions(snake_dict, piece_dict): 
        ... 
    def get_placement_actions(snake_dict, piece_dict): 

        ...
    def get_movement_actions(snake_pieces, piece_dict, enemy_pieces_dict): 
        movements = []
        for piece in snake_pieces: 
            if len(Piece.get_connections(piece, snake_pieces)) == 1: 
                legal_movement_positions = [pos for pos in Snake.get_perimeter(snake_pieces) if pos != piece and piece_dict[piece][0] >= enemy_pieces_dict[pos][0]]
                movements.extend(Actions.get_movement_codes(piece, legal_movement_positions))
        return movements  

    def get_perimeter(snake_pieces): 
        return list(set(chain.from_iterable([Piece.get_non_connections(piece, snake_pieces) for piece in snake_pieces]))) 

    def get_empty_perimeter(snake_pieces, enemy_pieces): 
        return list(set(chain.from_iterable([Piece.get_empty_adjacent(piece, snake_pieces, enemy_pieces) for piece in snake_pieces]))) 

    def add_piece_to_dict(snake_dict, piece_dict, pos, val): 
        connected_snakes = Piece.get_connected_snakes(pos, piece_dict) 
        print(f"connected_snakes: {connected_snakes}")

        #create parent snake 
        snake_id = Snake.get_unique_id() if not connected_snakes else Snake.get_combined_snakes(connected_snakes, snake_dict, piece_dict) 

        #add piece to the snake and piece dictionaries
        snake_dict[snake_id].append(pos)
        piece_dict[pos].extend([val, True, snake_id])

    def remove_piece_from_dict(snake_dict, piece_dict, pos): 
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
    
    def is_active(pieces): 
        return sum([data[1] for data in pieces.values()]) < 2 
    
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
    
    def default_value():
        return [-1, -1, -1]
    
class Actions:

    store_input = None

    def click_action(event):
        cell_pos = Game.coords_to_grid_pos(event.pos)
        state_indx = Game.pos_to_grid_index(cell_pos)

        if event.button == 1: 
            if Game.cell_empty(state_indx): 
                Game.active_player.run_action(Actions.get_placement_code(cell_pos))
            else: 
                Game.active_player.run_action(Actions.translate_roll(selected_cell.pos))
                
        if event.button == 3: 
            if Actions.store_input: 
                Game.run_action(Actions.translate_movement(Actions.store_input, selected_cell.pos))
                Actions.store_input = None 
            elif selected_cell.get_player() == Game.active_player: 
                Actions.store_input = selected_cell.pos

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
        return [Actions.get_movement_code(to_move.pos, loc.pos) for loc in locations]
    
    def get_placement_code(pos): 
        return "p"+"-"+str(pos[0])+"-"+str(pos[1])

    def get_placement_codes(cells): 
        return [Actions.get_placement_code(cell.pos) for cell in cells]

    def get_roll_code(pos): 
        return "r"+"-"+str(pos[0])+"-"+str(pos[1])

    def get_roll_codes(cells):
        return [Actions.get_roll_code(cell.pos) for cell in cells]

    def display():
        mouse_pos = pygame.mouse.get_pos()
        if Actions.store_input: 
            moving_cell = Game.grid_search(Actions.store_input) 
            pygame.draw.rect(Game.display_screen, moving_cell.color, (mouse_pos[0]-12.5, mouse_pos[1]-12.5, moving_cell.w/2, moving_cell.w/2), width=1)        
    
                    
        
        
        
        


        
        
        
        