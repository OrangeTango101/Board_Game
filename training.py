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
                    0: {"spawn_pos": (5,10), "num_pieces": 6, "num_placements": 3, "snake_dict": defaultdict(list), "piece_dict": defaultdict(list)},
                    1: {"spawn_pos": (5,0), "num_pieces": 6, "num_placements": 3, "snake_dict": defaultdict(list), "piece_dict": defaultdict(list)} 
                 }
    sim_players = [ReinforcementAgent(0, (255, 0, 0), "Red", model2, [ReinforcementAgent.win_bias]), ReinforcementAgent(1, (0, 255, 0), "Green", model1, [ReinforcementAgent.win_bias, ReinforcementAgent.dist_bias])]
    Game.initialize_game(sim_players, game_state)


def train_agents(winner):
    reinforcementAgents = [agent for agent in Game.players if isinstance(agent, ReinforcementAgent)]
    if winner: 
        print(f"Iterations:{40000-iterations} Epsilon:{ReinforcementAgent.epsilon}")

    if ReinforcementAgent.epsilon < 0.9: 
        ReinforcementAgent.epsilon *= 1.0005

    for agent in reinforcementAgents: 
        episode_value = 0
        if agent == winner: 
            episode_value = 1 
        elif winner: 
            episode_value = -1 
        #episode_value = 1 if agent == winner else 0
        agent.train(None, episode_value)
        if agent == winner: 
            Game.get_other_player(agent.id).train(ReinforcementAgent.flip_episode_perspective(agent.episode), episode_value)
        
        

start_new_game()    

while running:
    User.register_events()

    if User.close_game: 
        running = False
    
    if Game.winner:
        iterations -= 1 
        train_agents(Game.winner)
        start_new_game()

    if Game.rounds == 100:
        iterations -= 1 
        train_agents(None)
        start_new_game()

    if iterations == 0: 
        running = False

    Game.game_loop()

pygame.quit()
sys.exit()




