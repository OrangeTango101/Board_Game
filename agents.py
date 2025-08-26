from game import *
from model import *
import torch
import numpy as np
from time import sleep

class Agent(Player): 
    action_delay = 0

    def choose_action(self): 
        ... 
    
class RandomAgent(Agent): 
        
    def choose_action(self, game_state): 
        legal_actions = game_state.get_actions(self.id)
        action_types = [key for key in legal_actions.keys() if legal_actions[key]]
        chosen_type = legal_actions[np.random.choice(action_types)]
        game_state.run_action(self.id, np.random.choice(chosen_type))
        
class ReinforcementAgent(Agent):
    epsilon = 0.5 

    def __init__(self, id, color, name, model, training=None): 
        super().__init__(id, color, name)
        self.training = training
        self.episode = []
        self.nn = model
     
    def choose_action(self, game_state): 
        legal_actions = game_state.get_actions(self.id)
        highest_rating = float('-inf')
        best_action = None
        
        if np.random.rand() > ReinforcementAgent.epsilon: 
            action_types = [key for key in legal_actions.keys() if legal_actions[key]]
            actions = legal_actions[np.random.choice(action_types)]
            best_action = np.random.choice(actions)
        else: 
            for ls in legal_actions.values(): 
                for action in ls: 
                    rating = 0
                    if action.split("-")[0] == "r": 
                        projections = [game_state.generate_successor(self.id, droll).get_board_piece_state() for droll in Actions.get_droll_codes(action)]
                        rating = np.mean([self.nn(torch.tensor([projection], dtype=torch.float)).item() for projection in projections]) 
                    else: 
                        projection = game_state.generate_successor(self.id, action).get_board_piece_state()
                        rating = self.nn(torch.tensor([projection], dtype=torch.float)).item()
                    
                    if rating > highest_rating: 
                        highest_rating = rating
                        best_action = action 

        game_state.run_action(self.id, best_action)
        self.episode.append(game_state.get_board_piece_state())

    def train(self, episode, episode_value): 
        ...

        
    






        


     
         


