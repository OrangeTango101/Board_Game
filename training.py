import pygame
import sys

from game import Game
from game import Player
from user import User 
from agents import RandomAgent

iterations = 5
running = True

pygame.init()
Game.initialize_game(players=[Player(0, (5,10), 6, (255, 0, 0), "Red"), Player(0, (5,0), 6, (0, 255, 0), "Green")])

def train_agents(winner):
    ... 

while running:
    User.register_events()

    if User.close_game: 
        running = False
    
    if Game.winner:
        iterations -= 1 
        train_agents(Game.winner)
        Game.initialize_game(players=[Player(0, (5,10), 6, (255, 0, 0), "Red"), Player(0, (5,0), 6, (0, 255, 0), "Green")])

    if Game.rounds == 100: 
        iterations -= 1 
        train_agents(None)
        Game.initialize_game(players=[Player(0, (5,10), 6, (255, 0, 0), "Red"), Player(0, (5,0), 6, (0, 255, 0), "Green")])

    if iterations == 0: 
        running = False


    Game.display_game()

pygame.quit()
sys.exit()














'''
import pygame
import sys
from game import Game

class TrainingGame(Game):
    winners = []

    def initialize_game(players=[], iterations=100): 
        Game.initialize_game(players) 
        TrainingGame.iterations = iterations

    def test_over():
        if sum([player.active for player in Game.players]) == 1:
            TrainingGame.iterations -= 1 

        if TrainingGame.iterations == 0: 
            print(f"Training Over, Winners: {TrainingGame.winners}")  
            pygame.quit()
            sys.exit()
'''




