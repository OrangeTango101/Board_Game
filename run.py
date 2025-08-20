import pygame
import sys

from game import Game
from game import Player
from user import * 
from agents import *

two_players = [Player(0, (5,10), 6, (255, 0, 0), "Red"), Player(0, (5,0), 6, (0, 255, 0), "Green")]
three_players = [Player(0, (5,10), 6, (255, 0, 0), "Red"), Player(0, (5,0), 6, (0, 255, 0), "Green"), Player(0, (10,5), 6, (255, 0, 255), "Purple")]

pygame.init()
Game.initialize_game(players=[Player(0, (5,10), 6, (255, 0, 0), "Red"), Player(0, (5,0), 6, (0, 255, 0), "Green")])

running = True
while running:
    User_Testing.register_events()

    #Game.game_loop()

    if User.close_game or Game.winner: 
        running = False

    Game.game_state = Game.get_game_state(Game.players[0].pieces, Game.players[1].pieces)
    Game.display_game()

pygame.quit()
sys.exit()

