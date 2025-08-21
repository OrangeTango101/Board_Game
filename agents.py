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

    def __init__(self, id, spawn_pos, num_pieces, color, name, model, training=None): 
        super().__init__(id, spawn_pos, num_pieces, color, name)
        self.training = training
        self.episode = []
        self.nn = model

        self.epsilon = 0.8
     
    def choose_action(self, game_state): 
        legal_actions = Player.get_actions(game_state)
        highest_rating = float('-inf')
        best_action = None
        best_projection = None
        
        '''
        if np.random.rand() > self.epsilon: 
            action_types = [key for key in legal_actions.keys() if legal_actions[key]]
            actions = legal_actions[np.random.choice(action_types)]

            Player.run_action(np.random.choice(actions), game_state, legal_actions)
            return 
        '''


        for ls in legal_actions.values(): 
            for action in ls: 
                projection = ReinforcementAgent.project_state(action, game_state)
                rating = self.nn(torch.tensor([projection], dtype=torch.float)) 
                
                if rating > highest_rating: 
                    highest_rating = rating
                    best_action = action 
                    best_projection = projection


        self.episode.append([Game.get_board_state(game_state["piece_dict"], game_state["op_piece_dict"]), 0, best_projection])
        Player.run_action(best_action, game_state, legal_actions)
    
    def project_state(action_code, game_state):
        state = Game.get_board_state(game_state["piece_dict"], game_state["op_piece_dict"])

        print(state)
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






        


     
         


