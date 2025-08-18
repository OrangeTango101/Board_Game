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
    game_state = []
    show_display = True

    player_turn = 0

    def initialize_game(players=[]): 
        Game.game_state = Game.get_game_state(Game.players[0].snakes, Game.players[1].snakes)
        Game.display_screen = pygame.display.set_mode((Game.display_width, Game.display_height))
        Game.value_font = pygame.font.SysFont(None, 20)
        Game.text_font = pygame.font.SysFont(None, 40)
        Game.text_font2 = pygame.font.SysFont(None, 30)

        Game.rounds = 0
        Game.winner = None
        Game.players = players
        Game.player_turn = 0 
        Game.active_player = Game.get_active_player()
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

    def run_action(action_code): 
        action = action_code.split("-")
        action_type = action[0]

        if action_code in Game.legal_actions[action_type]: 
            if action_type == "p": 
                Game.add_piece_to_grid(Game.active_player, (int(action[1]), int(action[2])), 1)
                Game.active_player.num_pieces -= 1
            if action_type == "m":
                moved_value = Game.grid_search((int(action[1]), int(action[2]))).value
                Game.remove_piece_from_grid((int(action[1]), int(action[2])))
                Game.remove_piece_from_grid((int(action[3]), int(action[4])))
                Game.add_piece_to_grid(Game.active_player, (int(action[3]), int(action[4])), moved_value)
                Game.grid_search((int(action[3]), int(action[4]))).snake.moved()
            if action_type == "r": 
                cell_to_roll = Game.grid_search((int(action[1]), int(action[2]))) 
                cell_to_roll.roll()   
            
            Game.legal_actions = Game.active_player.get_actions()
            Game.game_state = Game.get_game_state(Game.players[0].snakes, Game.players[1].snakes)

    def add_piece_to_grid(player, pos, value): 
        new_piece = Game.grid_search(pos) 
        connected_cells = new_piece.get_connections(player)
        connected_snakes = list(set([cell.snake for cell in connected_cells])) 

        #update the perimeter of surrounding cells 
        for cell in connected_cells: 
            cell.remove_perimeter(new_piece)  

        #create parent snake 
        parent_snake = Snake(player, []) if not connected_snakes else Snake.get_combined_snakes(connected_snakes) 

        #place piece and add it to parent snake 
        new_piece.place_piece(parent_snake, value) 
        parent_snake.add_cell(new_piece)
         
                                        
    def remove_piece_from_grid(pos): 
        removed_piece = Game.grid_search(pos)
        player = removed_piece.get_player()

        if removed_piece.is_empty():
            return False 
        
        connected_cells = removed_piece.get_connections(player)
        removed_snake = removed_piece.snake

        #update the perimeter of surrounding cells
        for cell in connected_cells: 
            cell.add_perimeter(removed_piece)

        #remove piece 
        removed_piece.remove_piece()

        #handle broken snakes  
        if len(connected_cells) > 1: 
            while connected_cells: 
                snake_cells = []
                connected_cells[0].snake_search(snake_cells, connected_cells) 
                Snake(player, snake_cells)  

            removed_snake.delete_self()
        else: 
            removed_snake.remove_cell(removed_piece) 

        return True 

    def valid_search_pos(pos): 
        return pos[0] >= 0 and pos[0] <= Game.grid_width-1 and pos[1] >= 0 and pos[1] <= Game.grid_height-1

    def grid_search(pos, grid): 
        return grid[Game.pos_to_grid_index(pos)]
    
    def pos_to_grid_index(pos): 
        return pos[0]*Game.grid_height+pos[1]

    def get_selected_cell(mouse_pos):  
        for cell in Game.grid: 
            display_range = cell.get_display_range()
            if (mouse_pos[0] >= display_range[0][0] and mouse_pos[0] <= display_range[0][1] 
            and mouse_pos[1] >= display_range[1][0] and mouse_pos[1] <= display_range[1][1]):
                return cell

    def get_active_player(): 
        return Game.players[Game.player_turn]
    
    def get_inactive_player(): 
        return Game.players[(Game.player_turn+1)%2 ]

    def get_player(index):
        return Game.players[index]
    
    def get_game_state(p1_pieces, p2_pieces): 
        grid = [0]*(Game.grid_width*Game.grid_height)

        for piece, data in p1_pieces.items(): 
            grid[Game.pos_to_grid_index(piece)] = data[0] 
        for piece, data in p2_pieces.items(): 
            grid[Game.pos_to_grid_index(piece)] = -data[0]
        return grid


    def print_state(state): 
        rows = 11
        cols = len(state) // rows

        grid = [[0] * cols for _ in range(rows)]

        for i, num in enumerate(state):
            row = i % rows
            col = i // rows
            grid[row][col] = num

        for row in grid:
            print(" ".join(f"{num:2}" for num in row))

    def display_game():   
        Game.display_screen.fill((255, 255, 255))
        for player in Game.players:
            player.display()

        for cell in Game.game_state: 
            ...
        
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
        self.snakes = {}

        #{(piece_coords): (value, is_active, snake_id)}
        #pieces = {(0,1): (6,True,1), (2,2): (6,True,1), (3,4): (6,True,1), (5,5): (6,True,2), (9,7): (6,True,2), ...}
        self.pieces = {}

    def choose_action(self): 
        ...
        
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

    #Snake methods
    def remove_snake(self, snake):
        ... 

    #Piece methods 
    def roll_piece(self, pos): 
        ...
    def move_piece(self, pos1, pos2): 
        ... 
    def place_piece(self, pos): 
        ...
    def activate_pieces(self): 
        ... 
    def delete_piece(self, pos): 
        ...
    def delete_all_pieces(self): 
        ... 
    def total_pieces(self): 
        ... 
    
    def display(self): 
        pygame.draw.circle(Game.display_screen, self.color, (self.spawn_pos[0]*50+25, self.spawn_pos[1]*50+25), 20, width=1)


# TODO: Make Snake class static such that it returns legal actions given a snake dictionary 

class Snake: 

    def get_actions(snake): 
        ... 

    def get_legal_movements(self):
        actions = []
        if not self.active or self.rolls != 1: 
            return []
            
        for tail in self.get_tails(): 
            perimeter = list(set(chain.from_iterable([cell.perimeter for cell in self.cells if cell != tail]))) 
            legal_moves = [cell for cell in perimeter if cell.value <= tail.value]
            actions.extend(Actions.translate_movement_batch(tail, legal_moves))   
            
        return actions

    def get_legal_rolls(self): 
        if not self.active: 
            return [] 
        return Actions.translate_roll_batch([cell for cell in self.cells if not cell.rolled])

    def add_roll(self, val): 
        if self.rolls == 1 or len(self.cells) == 1: 
            self.rolls = 0 
            self.active = False
        else: 
            self.rolls += 1 

        self.check_shared_vals()

    def check_shared_vals(self):    
        shared_val = True
        val = self.cells[0].value
        for cell in self.cells:
            if cell.value != val or cell.value == 1: 
                shared_val = False
        if shared_val and len(self.cells) > 1: 
            for cell in self.cells:
                cell.value = 1
            self.active = False 
            self.player.num_pieces += 1

    def moved(self): 
        self.active = False
    
    def get_tails(self): 
        return [cell for cell in self.cells if len(cell.get_connections(self.player)) == 1] 

    def get_perimeter(self): 
        return list(set(chain.from_iterable([cell.perimeter for cell in self.cells]))) 

    def get_empty_perimeter(self): 
        perim = []
        for cell in self.cells: 
            perim.extend([perim_cell for perim_cell in cell.perimeter if perim_cell.is_empty()])
        return perim

    def add_cell(self, cell): 
        self.cells.append(cell) 

    def remove_cell(self, cell): 
        self.cells.remove(cell) 
        if not self.cells: 
            self.delete_self()

    def reset(self):
        self.active = True
        self.rolls = 0
        self.check_shared_vals()
        for cell in self.cells:
            cell.rolled = False

    def delete_self(self, keep_cells=True): 
        for cell in self.cells: 
            if cell.snake == self: 
                if keep_cells: 
                    cell.snake = None
                else: 
                    cell.remove_piece()
        self.player.remove_snake(self) 
        
    def get_combined_snakes(snakes):
        if not snakes: 
            return None 
        if len(snakes) == 1:
            return snakes[0]
        combined_cells = list(chain.from_iterable([snake.cells for snake in snakes]))
        player = snakes[0].player 
        active = all([snake.active for snake in snakes]) 
        rolls = max([snake.rolls for snake in snakes]) 

        for snake in snakes: 
            snake.delete_self() 
        
        return Snake(player, combined_cells, active, rolls) 
    
    def get_perim(snake):
        ...

    def is_active(pieces, player): 
        ... 

class Piece:
    rel_edge_positions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    def get_connections(pos, pieces): 
        if pos not in pieces: 
            return[]
        return [(pos[0]+rel_edge[0], pos[1]+rel_edge[1]) for rel_edge in Piece.rel_edge_position if Game.valid_search_pos((pos[0]+rel_edge[0], pos[1]+rel_edge[1])) and (pos[0]+rel_edge[0], pos[1]+rel_edge[1]) in pieces]
        
    def get_non_connections(pos, pieces):
        if pos not in pieces: 
            return Piece.get_adjacent(pos)
        return [(pos[0]+rel_edge[0], pos[1]+rel_edge[1]) for rel_edge in Piece.rel_edge_position if Game.valid_search_pos((pos[0]+rel_edge[0], pos[1]+rel_edge[1])) and (pos[0]+rel_edge[0], pos[1]+rel_edge[1]) not in pieces]

    def get_adjacent(pos): 
        return [(pos[0]+rel_edge[0], pos[1]+rel_edge[1]) for rel_edge in Piece.rel_edge_position if Game.valid_search_pos((pos[0]+rel_edge[0], pos[1]+rel_edge[1]))]
        
    def snake_search(self, visited, to_find): 
        visited.append(self)
        if self in to_find: 
            to_find.remove(self) 
        for connection in self.get_connections(self.get_player()): 
            if connection not in visited: 
                connection.snake_search(visited, to_find) 
        
class Actions:

    store_input = None

    def click_action(event):
        click_pos = event.pos
        selected_cell = Game.get_selected_cell(click_pos)

        if event.button == 1: 
            
            if selected_cell.is_empty(): 
                Game.run_action(Actions.translate_placement(selected_cell.pos))
            else: 
                Game.run_action(Actions.translate_roll(selected_cell.pos))
                
        if event.button == 3: 
            if Actions.store_input: 
                Game.run_action(Actions.translate_movement(Actions.store_input, selected_cell.pos))
                Actions.store_input = None 
            elif selected_cell.get_player() == Game.active_player: 
                Actions.store_input = selected_cell.pos

    def space_action(event): 
        Game.show_display = not Game.show_display 
    
    def translate_movement(pos1, pos2): 
        return "m"+"-"+str(pos1[0])+"-"+str(pos1[1])+"-"+str(pos2[0])+"-"+str(pos2[1])

    def translate_movement_batch(to_move, locations):
        return [Actions.translate_movement(to_move.pos, loc.pos) for loc in locations]
    
    def translate_placement(pos): 
        return "p"+"-"+str(pos[0])+"-"+str(pos[1])

    def translate_placement_batch(cells): 
        return [Actions.translate_placement(cell.pos) for cell in cells]

    def translate_roll(pos): 
        return "r"+"-"+str(pos[0])+"-"+str(pos[1])

    def translate_roll_batch(cells):
        return [Actions.translate_roll(cell.pos) for cell in cells]

    def display():
        mouse_pos = pygame.mouse.get_pos()
        if Actions.store_input: 
            moving_cell = Game.grid_search(Actions.store_input) 
            pygame.draw.rect(Game.display_screen, moving_cell.color, (mouse_pos[0]-12.5, mouse_pos[1]-12.5, moving_cell.w/2, moving_cell.w/2), width=1)        
    
                    
        
        
        
        


        
        
        
        