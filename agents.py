from game import *
from time import sleep

class Agent(Player): 
    action_delay = 1

    def take_turn(self): 
        while Game.active_player == self: 
            Game.try_action(self.choose_action()) 
            sleep(Agent.action_delay)
            Game.display_game()

    def choose_action(self): 
        ... 
    
class RandomAgent(Agent): 
        
    def choose_action(self): 
        action_types = [key for key in Game.legal_actions.keys() if Game.legal_actions[key]]
        actions = Game.legal_actions[np.random.choice(action_types)]

        return np.random.choice(actions)

         
class ReinforcementAgent(Agent):
     
     def choose_action(self): 
          ... 
         


