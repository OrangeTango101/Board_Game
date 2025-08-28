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

    def __init__(self, id, color, name, model, biases, training=None): 
        super().__init__(id, color, name)
        self.training = training
        self.episode = []
        self.nn = model
        self.biases = biases
     
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
                        projections = [game_state.generate_successor(self.id, droll) for droll in Actions.get_droll_codes(action)]
                        rating = np.mean([self.state_eval(projection) for projection in projections]) 
                    else: 
                        projection = game_state.generate_successor(self.id, action)
                        rating = self.state_eval(projection)
                    
                    if rating > highest_rating: 
                        highest_rating = rating
                        best_action = action 

        game_state.run_action(self.id, best_action)
        self.episode.append(game_state.get_copy())
        
    def train(self, episode, episode_value): 
        if not episode: 
            episode = self.episode
        loss = nn.MSELoss()

        for indx, game_state in enumerate(episode):
            state_tensor = torch.tensor([game_state.get_board_piece_state()], dtype=torch.float)
 
            target = max(episode_value*0.3, episode_value*(indx/len(episode)), key=abs)
            prediction = self.nn(state_tensor)

            self.nn.optimizer.zero_grad() 
            l = loss(prediction, torch.tensor([[target]], dtype=torch.float)) 
            l.backward()

            self.nn.optimizer.step()


    def state_eval(self, game_state): 
        board_state = game_state.get_board_piece_state()

        return self.nn(torch.tensor([board_state], dtype=torch.float)).item()+sum([bias(self.id, game_state) for bias in self.biases])
    
    def win_bias(player, game_state): 
        if game_state.get_winner() and game_state.get_winner() == player: 
            return 100000
        else: 
            return 0

    def dist_bias(player, game_state): 
        distance_bias = 1

        closest_dist = 1000
        for piece in game_state[player]["piece_dict"]: 
            enemy_spawn = game_state[(player+1)%2]["spawn_pos"]
            dist = abs(enemy_spawn[0]-piece[0])+abs(enemy_spawn[1]-piece[1])
            if dist < closest_dist: 
                closest_dist = dist
        distance_bias += 1/max(closest_dist, 1)

        return distance_bias

    def flip_episode_perspective(episode): 
        for game_state in episode: 
            game_state.flip_perspective()
        return episode



    '''
    def flip_episode_perspective(episode): 
        flipped_episode = []
        x_line = (Game.grid_height//2) 
        y_line = (Game.grid_width//2)

        for board_piece_state in episode: 
            flipped_board = board_piece_state[:-2][::-1]
            flipped_pieces = board_piece_state[-2:][::-1]
            flipped_board_piece_state = flipped_board+flipped_pieces
            flipped_episode.append(flipped_board_piece_state)
        return flipped_episode
    '''
            

        
    






        


     
         


