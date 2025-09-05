import pygame
import sys
from collections import defaultdict

from game import Game
from game import Player
from user import *  
from agents import *

'''
When the run file is run it will start the game. It will continue running until a player wins. 
A user can customize the color and name arguments of each player, but should not change the id value. 
'''

pygame.init()
player0 = Player(
    id = 0, #should not be changed 
    color = (255, 0, 0),
    name = "Red"
)
player1 = Player(
    id = 1, #should not be changed 
    color = (0, 255, 0),
    name = "Green"
)
players = [player0, player1]

Game.initialize_game(players)

running = True
while running:
    User.register_events()
    Game.game_loop()

    if User.close_game or Game.winner: 
        running = False
    
pygame.quit()
sys.exit()

