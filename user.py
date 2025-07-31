import pygame
from game import Actions

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
                    print("Window focused — next click will be ignored")
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if User.ignore_next_click: 
                    print("Ignoring first click after focus")
                    User.ignore_next_click = False
                else: 
                    Actions.click_action(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    Actions.space_action(event)

