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

                    player = Game.players[0]
                    enemy = Game.players[1]
                    game_state = Game.get_current_game_state()

                    Snake.add_piece_to_dict(cell_pos, 1, game_state)

                    Player.get_actions(player.num_pieces, player.spawn_pos, player.snakes, player.pieces, enemy.pieces)
                    print("add_success")
                    print(f"snakes: {player.snakes}")
                    print(f"pieces: {player.pieces}")
                    print("")

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    cell_pos = Game.coords_to_grid_pos(pygame.mouse.get_pos())
                    player = Game.active_player
                    game_state = Game.get_current_game_state()

                    Snake.remove_piece_from_dict(cell_pos, game_state)
                    print("remove_success")
                    print(f"snakes: {player.snakes}")
                    print(f"pieces: {player.pieces}")
                    print("")