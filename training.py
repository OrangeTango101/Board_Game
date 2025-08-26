import pygame
import sys

from game import Game
from game import Player
from user import User 
from agents import *

iterations = 40000
running = True

pygame.init()

model1 = SimpleGameNN()
model2 = SimpleGameNN()

def start_new_game():   
    game_state = {
                    0: {"spawn_pos": (5,8), "num_pieces": 2, "snake_dict": defaultdict(list), "piece_dict": defaultdict(list)},
                    1: {"spawn_pos": (5,2), "num_pieces": 2, "snake_dict": defaultdict(list), "piece_dict": defaultdict(list)} 
                 }
    sim_players = [ReinforcementAgent(0, (255, 0, 0), "Red", model2), ReinforcementAgent(1, (0, 255, 0), "Green", model1)]
    Game.initialize_game(sim_players, game_state)

loss = nn.MSELoss()
a = 0.01

def train_agents(winner):
    reinforcementAgents = [agent for agent in Game.players if isinstance(agent, ReinforcementAgent)]

    for agent in reinforcementAgents: 
        episode_value = 0
        if winner == agent: 
            episode_value = 1
            print(f"Iterations:{40000-iterations}")
        elif winner: 
            episode_value = 0

        for indx, state in enumerate(agent.episode): 
            state_tensor = torch.tensor([state], dtype=torch.float)

            target = max(episode_value*0.3, episode_value*(indx/len(agent.episode)))
            prediction = agent.nn(state_tensor)

            agent.nn.optimizer.zero_grad() 
            l = loss(prediction, torch.tensor([[target]], dtype=torch.float)) 
            l.backward()

            agent.nn.optimizer.step()
        if ReinforcementAgent.epsilon < 0.9: 
            ReinforcementAgent.epsilon *= 1.000005

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
        #train_agents(None)
        start_new_game()

    if iterations == 0: 
        running = False

    Game.game_loop()

pygame.quit()
sys.exit()




