import pygame
import sys

from game import Game
from game import Player
from user import User 
from agents import *

iterations = 40000
running = True

pygame.init()

'''
highly trained offense: "model_1_20250828-214128.pth"
'''

model1 = SimpleGameNN()
model2 = SimpleGameNN()
epsilon1 = 0.7
epsilon2 = 0.7

def start_new_game():   
    sim_players = [ReinforcementAgent(0, (255, 0, 0), "Red", model1, [ReinforcementAgent.win_bias], epsilon1, True), ReinforcementAgent(1, (0, 255, 0), "Green", model2, [ReinforcementAgent.win_bias], epsilon2, False, "model_1_20250828-214128.pth")]
    Game.initialize_game(sim_players)


def train_agents(winner):
    reinforcementAgents = [agent for agent in Game.players if isinstance(agent, ReinforcementAgent)]
    if winner: 
        print(f"Iterations:{40000-iterations} Epsilons:{epsilon1}, {epsilon2}")

    for agent in reinforcementAgents: 
        episode_value = 0
        if agent == winner: 
            episode_value = 1 
        elif winner: 
            episode_value = -1 
        if agent._train: 
            agent.train(None, episode_value)

    '''
    if winner: 
        loser = Game.get_other_player(winner.id)
        if loser in reinforcementAgents: 
            loser.train(winner.episode, 1)
    '''

        
start_new_game()    
while running:
    User.register_events()

    if User.close_game: 
        running = False
    
    if Game.winner:
        iterations -= 1 
        epsilon1 = epsilon1 * 1.00005 if epsilon1 < 0.9 else epsilon1
        epsilon2 = epsilon2 * 1.00005 if epsilon2 < 0.9 else epsilon2

        train_agents(Game.winner)
        start_new_game()

    if Game.rounds == 100:
        iterations -= 1 
        epsilon1 = epsilon1 * 1.00005 if epsilon1 < 0.9 else epsilon1
        epsilon2 = epsilon2 * 1.00005 if epsilon2 < 0.9 else epsilon2
        #train_agents(None)
        start_new_game()

    if iterations == 0: 
        running = False

    Game.game_loop()

pygame.quit()
sys.exit()




