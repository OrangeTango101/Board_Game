import pygame
import sys

from game import *
from user import User 
from agents import *

'''
When the training file is run it will continue running games until it has reached train_length number of games. 
A user can control the train_length, game_length, and agents in the Train class below. 
'''

class Train:
    train_length = 40000 #in games 
    game_length = 100 #in rounds 

    p0_wins = 0
    p1_wins = 0 

    models = [SimpleGameNN(), SimpleGameNN()] 
    epsilons = [0.6, 0.6]

    def start_new_game(train=True):  
        '''
        This method starts a new game and decides to whether to train agents based on the train argument. 
        The effects of the ReinforcementAgent args can be found in the training.py file. 
        Args: 
            train (bool): whether to train the agents 
        '''

        reinforcement_agent0 = ReinforcementAgent(
            id = 0, #should not be changed 
            color = (255, 0, 0), 
            name = "red", 
            model = Train.models[0], #should not be changed 
            biases = [ReinforcementAgent.win_bias, ReinforcementAgent.dist_bias], 
            epsilon = Train.epsilons[0],
            _train = False,
            enemy_learning = False, 
            training_file = None
        )
        reinforcement_agent1 = ReinforcementAgent(
            id = 1, #should not be changed 
            color = (0, 255, 0),
            name = "green", 
            model = Train.models[1], #should not be changed 
            biases = [ReinforcementAgent.win_bias, ReinforcementAgent.dist_bias], 
            epsilon = Train.epsilons[1],
            _train = False,
            enemy_learning = False, 
            training_file = None
        )

        Train.train_length -= 1 
        Train.epsilons[0] = Train.epsilons[0] * 1.00005 if Train.epsilons[0] < 0.9 else Train.epsilons[0]
        Train.epsilons[1] = Train.epsilons[1] * 1.00005 if Train.epsilons[1] < 0.9 else Train.epsilons[1]

        if Game.winner: 
            if Game.winner == Game.players[0]:
                Train.p0_wins += 1 
            if Game.winner == Game.players[1]:
                Train.p1_wins += 1 
            iterations = 40000-Train.train_length
            print(f"Iterations:{iterations} p0:{Train.p0_wins} p1:{Train.p1_wins} Epsilons:{Train.epsilons[0]}, {Train.epsilons[1]}")

        if train: 
            Train.train_agents(Game.winner)

        sim_players = [reinforcement_agent0, reinforcement_agent1]
        Game.initialize_game(sim_players)

    def train_agents(winner):
        reinforcementAgents = [agent for agent in Game.players if isinstance(agent, ReinforcementAgent)]

        for agent in reinforcementAgents: 
            episode_value = 0
            if agent == winner: 
                episode_value = 1 
            elif winner: 
                episode_value = -1
            if agent._train: 
                agent.train(None, episode_value, agent.id)
        
        if winner: 
            loser = Game.get_other_player(winner.id)
            if loser in reinforcementAgents and loser._train and loser.enemy_learning: 
                loser.train(winner.episode, 1, winner.id)

pygame.init()
running = True
Train.start_new_game(train=False)    

while running:
    User.register_events()

    if User.close_game: 
        running = False
    
    if Game.winner:
        Train.start_new_game()

    if Game.rounds == Train.game_length:
        Train.start_new_game()

    if Train.train_length == 0: 
        running = False

    Game.game_loop()

pygame.quit()
sys.exit()




