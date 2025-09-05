
<h1>Making A Board Game</h1>

<h2>Description</h2>
The project consists of a 2-player chance/strategy board game I invented. The game is played on an 11x11 board and the objective is to occupy the opponent's spawn on the other end of the board. A player accomplishes this by placing, moving, and rolling their pieces through the rules dictated below. A user can additionally train and play against reinforcment learning agents using a pytorch nn model.

<h2>How To Use The Program</h2>
The file "run.py" will start an instance of the game when run and will terminate once a player wins. The file "training.py"
will run multiple games that can be used to train reinforcment learning agents.  

<h2>Languages And Utilities Used</h2>

- <b>Python</b> 
- <b>PyTorch</b>
- <b>NumPy</b>
- <b>Pygame</b>

<h2>Definitions, Rules, And Controls</h2>

<h3>Definitions</h3>

- <b>Piece Reserve</b>
  - The number of pieces a player has available to place
  - Each player starts with 6 pieces in reserve
 
- <b>Spawn</b>
  - The only positon on the board where a player can place their pieces
  - A spawn is occupied if a piece is located at its position 
  - A player wins by occupying the opponents spawn
  - Represented by a circle in the player's color
    
- <b>Piece</b>
  - The tokens each player can place, move, and roll 
  - A piece becomes innactive if it is rolled or is part of a snake that is moved 
  - Represented by the player's color and text indicating a value between 1-6
 
- <b>Snake</b>
  - A chain of adjacent pieces that all belong to the same player
  - The perimeter of a snake are all the grid positions that are immediately adjacent
  - A snake is innactive if two of its pieces are rolled or a piece is moved 
 
- <b>Spawn Snake</b>
  - A type of snake that has a piece located in its own player's spawn

<h3>Rules</h3>

- <b>Placements</b>
  - Pieces can be placed in a player's spawn or on the perimeter of a spawn snake (see definition above)
  - A player can only make 3 placements a turn
  - A player can only make placements if their piece reserve is > 0
 
- <b>Rolls</b>
  - Rolling a piece randomly selects its value between 1-6
  - A piece can only be rolled if it is active
 
- <b>Movements</b>
  - A piece can only be moved if it is adjacent to exactly one other piece
  - A piece can only be moved to a position in its snake's perimeter
  - A piece can only be moved if exactly one piece in its snake has been rolled
  - A piece can be moved onto an enemy piece only if its value is >= the enemy piece's value
 
<h3>Controls</h3>

- <b>Left Click</b>
  - Attempts to place a piece if the position in unoccupied
  - Attempts to roll a piece if the position is occupied
 
- <b>Right Click</b>
  - Attempts to select a piece to move if none has already been selected
  - Attempts to move a piece if one has already been selected
 
- <b>Left Arrow Key</b>
  - Will undo the last move 
 
- <b>Space Bar</b>
  - Will toggle the visual display during agent training
 
- <b>Up/Down Arrow Key</b>
  - Increases/Decreases the time delay between agent actions

- <b>0/1 Keys</b>
  - Will save agent0/agent1 training to the working directory  


<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
