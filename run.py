import pygame
import sys
from collections import defaultdict

from game import Game
from game import Player
from user import *  
from agents import *

pygame.init()
players = [Player(0, (255, 0, 0), "Red"), Player(1, (0, 255, 0), "Green")]

Game.initialize_game(players)

running = True
while running:
    User.register_events()
    Game.game_loop()

    if User.close_game or Game.winner: 
        running = False
    
pygame.quit()
sys.exit()

