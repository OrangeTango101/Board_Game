import pygame
import sys

from game import Game
from game import Player
from user import User 
from agents import *

iterations = 20000
running = True

pygame.init()

model1 = SimpleGameNN()
model2 = SimpleGameNN()

def start_new_game():   
    sim_players = [ReinforcementAgent(0, (5,10), 2, (255, 0, 0), "Red", model1), RandomAgent(0, (5,4), 2 , (0, 255, 0), "Green")]
    Game.initialize_game(players=sim_players)

loss = nn.MSELoss()
a = 0.01

def train_agents(winner):
    reinforcementAgents = [agent for agent in Game.players if isinstance(agent, ReinforcementAgent)]

    for agent in reinforcementAgents: 
        episode_value = 0 
        if winner == agent: 
            episode_value = 1
            print(f"Iterations:{20000-iterations}")
        elif winner: 
            episode_value = -1

        for state, reward, next_state in agent.episode: 
            state_tensor = torch.tensor([state], dtype=torch.float)
            next_tensor = torch.tensor([next_state], dtype=torch.float)

            target = episode_value
            prediction = agent.nn(state_tensor)

            agent.nn.optimizer.zero_grad() 
            l = loss(prediction, torch.tensor([[target]], dtype=torch.float)) 
            l.backward()

            agent.nn.optimizer.step()

start_new_game()

while running:
    User.register_events()

    if User.close_game: 
        running = False
    
    if Game.winner:
        iterations -= 1 
        train_agents(Game.winner)
        start_new_game()

    if Game.rounds == 30:
        iterations -= 1 
        train_agents(None)
        start_new_game()

    if iterations == 0: 
        running = False

    Game.game_loop()

    if Game.show_display: 
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




