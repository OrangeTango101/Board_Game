from game import *
from model import *
import torch
from time import sleep

class Agent(Player): 
    action_delay = 0

    def choose_action(self): 
        ... 
    
class RandomAgent(Agent): 
        
    def choose_action(self, game_state): 
        legal_actions = Player.get_actions(game_state)
        action_types = [key for key in legal_actions.keys() if legal_actions[key]]
        actions = legal_actions[np.random.choice(action_types)]

        Player.run_action(np.random.choice(actions), game_state, legal_actions)
        
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
        best_projection = None
        
        
        if np.random.rand() > ReinforcementAgent.epsilon: 
            action_types = [key for key in legal_actions.keys() if legal_actions[key]]
            actions = legal_actions[np.random.choice(action_types)]

            game_state.run_action(self.id, np.random.choice(actions))
            return 
        
        for ls in legal_actions.values(): 
            for action in ls: 
                projection = ReinforcementAgent.project_state(action, game_state)
                rating = self.nn(torch.tensor([projection], dtype=torch.float)) 
                
                if rating > highest_rating: 
                    highest_rating = rating
                    best_action = action 
                    best_projection = projection


        self.episode.append([game_state.get_board_state(), 0, best_projection])
        game_state.run_action(self.id, best_action)
    
    def project_state(action_code, game_state):
        board_state = game_state.get_board_state()

        action = action_code.split("-")
        action_type = action[0]

        if action_type == "p": 
            board_state[int(action[1])*Game.grid_height+int(action[2])] = 1
        if action_type == "m":
            board_state[int(action[3])*Game.grid_height+int(action[4])] = board_state[int(action[1])*Game.grid_height+int(action[2])]
            board_state[int(action[1])*Game.grid_height+int(action[2])] = 0
        if action_type == "r": 
            board_state[int(action[1])*Game.grid_height+int(action[2])]= 3.5

        return board_state






        


     
         


