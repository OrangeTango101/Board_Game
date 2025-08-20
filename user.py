import pygame
from game import *

class User: 
    close_game = False
    ignore_next_click = True

    store_input = None
    
    def register_events(): 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                User.close_game = True
                
            if event.type == pygame.ACTIVEEVENT:
                if event.gain == 1 and event.state == 1:  # Input focus gained
                    User.ignore_next_click = True
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if User.ignore_next_click: 
                    User.ignore_next_click = False
                else: 
                    Actions.click_action(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    Actions.space_action(event)

class User_Testing: 
    close_game = False
    ignore_next_click = True

    store_input = None
    
    def register_events(): 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                User.close_game = True
                
            if event.type == pygame.ACTIVEEVENT:
                if event.gain == 1 and event.state == 1:  # Input focus gained
                    User.ignore_next_click = True
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if User.ignore_next_click: 
                    User.ignore_next_click = False
                else: 
                    click_coords = event.pos
                    cell_pos = Game.coords_to_grid_pos(click_coords)

                    player = Game.active_player
                    Snake.add_piece_to_dict(player.snakes, player.pieces, cell_pos, 1)
                    print("add_success")
                    print(f"snakes: {player.snakes}")
                    print(f"pieces: {player.pieces}")
                    print("")
                    

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    cell_pos = Game.coords_to_grid_pos(pygame.mouse.get_pos())

                    player = Game.active_player
                    Snake.remove_piece_from_dict(player.snakes, player.pieces, cell_pos)
                    print("remove_success")
                    print(f"snakes: {player.snakes}")
                    print(f"pieces: {player.pieces}")
                    print("")