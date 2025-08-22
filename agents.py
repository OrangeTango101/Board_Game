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

'''
class MinimaxAgent(Agent):

    def __init__(self, id, spawn_pos, num_pieces, color, name, model, training=None): 
        super().__init__(id, spawn_pos, num_pieces, color, name)
        self.training = training
        self.episode = []
        self.nn = model

    def choose_action(self, game_state): 
        ...
    
    def minmax_search(self, agent, depthCount, gameState: GameState, eval_func): 

        totalAgents = gameState.getNumAgents()
        if agentIndex == totalAgents:
            agentIndex = 0
            depthCount += 1

        if gameState.isWin() or gameState.isLose() or depthCount == self.depth:   
            return [self.evaluationFunction(gameState), Directions.STOP] 

        actions = gameState.getLegalActions(agentIndex)
        chosen_action = Directions.STOP 
        if agentIndex == 0: 
            largest_val = -1e9 
            for action in actions:
                action_value = self.minmaxSearch(agentIndex+1, depthCount, gameState.generateSuccessor(agentIndex, action))
                if action_value[0] > largest_val: 
                    largest_val = action_value[0]
                    chosen_action = action
            return [largest_val, chosen_action]
        else:
            smallest_val = 1e9
            for action in actions: 
                action_value = self.minmaxSearch(agentIndex+1, depthCount, gameState.generateSuccessor(agentIndex, action))
                if action_value[0] < smallest_val: 
                    smallest_val = action_value[0]
                    chosen_action = action 
            return [smallest_val, chosen_action]
        
    def evaluation_function(game_state):
        ...
    
'''
        
class ReinforcementAgent(Agent):
    epsilon = 0.5 

    def __init__(self, id, spawn_pos, num_pieces, color, name, model, training=None): 
        super().__init__(id, spawn_pos, num_pieces, color, name)
        self.training = training
        self.episode = []
        self.nn = model
     
    def choose_action(self, game_state): 
        legal_actions = Player.get_actions(game_state)
        highest_rating = float('-inf')
        best_action = None
        best_projection = None
        
        
        if np.random.rand() > ReinforcementAgent.epsilon: 
            action_types = [key for key in legal_actions.keys() if legal_actions[key]]
            actions = legal_actions[np.random.choice(action_types)]

            Player.run_action(np.random.choice(actions), game_state, legal_actions)
            return 
        

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






        


     
         


