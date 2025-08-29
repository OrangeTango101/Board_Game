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
    
    saved_agents = [ReinforcementAgent(0, (255, 0, 0), "Red", model1, [ReinforcementAgent.win_bias, ReinforcementAgent.mobility_bias, ReinforcementAgent.dist_bias]), ReinforcementAgent(1, (0, 255, 0), "Green", model2, [ReinforcementAgent.win_bias, ReinforcementAgent.mobility_bias, ReinforcementAgent.dist_bias, ReinforcementAgent.piece_bias, ReinforcementAgent.snakes_bias])]

    sim_players = [Player(0, (255, 0, 0), "Red"), ReinforcementAgent(1, (0, 255, 0), "Green", model2, [ReinforcementAgent.win_bias], "model_1_20250828-214128.pth")]
    Game.initialize_game(sim_players, game_state)


def train_agents(winner):
    reinforcementAgents = [agent for agent in Game.players if isinstance(agent, ReinforcementAgent)]
    if winner: 
        print(f"Iterations:{40000-iterations} Epsilon:{ReinforcementAgent.epsilon}")

    for agent in reinforcementAgents: 
        episode_value = 0
        if agent == winner: 
            episode_value = 1 
        elif winner: 
            episode_value = -1 
        agent.train(None, episode_value)

    if winner: 
        loser = Game.get_other_player(winner.id)
        if loser in reinforcementAgents: 
            loser.train(ReinforcementAgent.flip_episode_perspective(winner.episode), 1)

        
start_new_game()    
while running:
    User.register_events()

    if User.close_game: 
        running = False
    
    if Game.winner:
        iterations -= 1 
        if ReinforcementAgent.epsilon < 0.9: 
            ReinforcementAgent.epsilon *= 1.00005
        train_agents(Game.winner)
        start_new_game()

    if Game.rounds == 50:
        iterations -= 1 
        if ReinforcementAgent.epsilon < 0.9: 
            ReinforcementAgent.epsilon *= 1.00005
        train_agents(None)
        start_new_game()

    if iterations == 0: 
        running = False

    Game.game_loop()

pygame.quit()
sys.exit()




