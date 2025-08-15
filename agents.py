from game import *
from model import *
import torch
from time import sleep

class Agent(Player): 
    action_delay = 0

    def choose_action(self): 
        ... 
    
class RandomAgent(Agent): 
        
    def choose_action(self): 
        action_types = [key for key in Game.legal_actions.keys() if Game.legal_actions[key]]
        actions = Game.legal_actions[np.random.choice(action_types)]

        Game.run_action(np.random.choice(actions))

class ReinforcementAgent(Agent):

    def __init__(self, id, spawn_pos, num_pieces, color, name, model, training=None): 
        super().__init__(id, spawn_pos, num_pieces, color, name)
        self.training = training
        self.episode = []
        self.nn = model

        self.epsilon = 0.8
     
    def choose_action(self): 
        highest_rating = float('-inf')
        best_action = None
        best_projection = None

        if np.random.rand() > self.epsilon: 
            action_types = [key for key in Game.legal_actions.keys() if Game.legal_actions[key]]
            actions = Game.legal_actions[np.random.choice(action_types)]

            Game.run_action(np.random.choice(actions))
            return 


        for ls in Game.legal_actions.values(): 
            for action in ls: 
                projection = self.project_state(action)
                rating = self.nn(torch.tensor([projection], dtype=torch.float)) 
                
                if rating > highest_rating: 
                    highest_rating = rating
                    best_action = action 
                    best_projection = projection


        self.episode.append([Game.get_game_state(self), 0, best_projection])
        Game.run_action(best_action)
    
    def project_state(self, action_code):
        state = Game.get_game_state(self)
        action = action_code.split("-")
        action_type = action[0]

        if action_type == "p": 
            state[int(action[1])*Game.grid_height+int(action[2])] = 1
        if action_type == "m":
            state[int(action[3])*Game.grid_height+int(action[4])] = state[int(action[1])*Game.grid_height+int(action[2])]
            state[int(action[1])*Game.grid_height+int(action[2])] = 0
        if action_type == "r": 
            state[int(action[1])*Game.grid_height+int(action[2])]= 3.5

        return state






        


     
         


