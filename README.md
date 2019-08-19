
# BASHGAMMON
	reengineed by Integral Team
	started by Jordan Balk Schaer
	
## Description

BASHGAMMON is a Python/text version of the classic game Backgammon.  It is played with the rules that I know.  
There are two teams, represented as X and O.  O starts in the upper right side and X starts in the lower right.  X wants to move to its home board in the upper right and O wants to move to its home board in the lower right.  
If the two die rolled are equal (are doubles) then you have twice as many moves, otherwise you have the sum of the two rolls to move.
If a piece is alone, the opposing team can jump it and send it to jail.  If a piece is in jail, you cannot move until you first free your piece from jail.
The team that gets all of its pieces off of the board first wins.

## How-to PLAY
- launch _backgammon.py_

Type in two numbers, the first number is the position of the piece you want to move, and the second is the number of spots you want to move it by; for example, if you were O and you typed (1,6) you would move a piece at position 1 6 spots.
You can type 1,6 or 1 6 but you cannot type 16, the computer won't know what you mean by that.
You can undo one move by typing "u" or "undo" but you cannot undo more than the previous move.
You can finish your turn if you type "f" or "finish" or "d" or "done".  Your turn will finish automatically if you run out of moves.

Currently, the available game types are :
- 'pvp': person Vs. person 
- 'pvr': person Vs. weak
- 'pvq': person Vs. medium  
- 'pvd': person Vs. hard 
- 'pvc': person Vs. nightmare 
- 'rvr': weak Vs. weak 
- 'qvr': medium Vs. weak 
- 'dvr': hard Vs. weak 
- 'cvr': nightmare Vs. weak 

## How-to TRAIN
- open _backgammon-player.ipynb_ in a Jupyter Notebook
- run the cell with imports and game functions
- go to the block Training /  Generating dataset for CNN
- set the _epochs_ parameter which represents the number of games to be generated
- launch _train.py_ once the _data/_ folder with images and csv files is generated

## CNN Architecture
- Input data
- input -> img(256, 128)  # Games board image with size 256х128
- input2 -> array(1)   # The value of the dice.
- input -> convolution 2d -> relu -> max pooling 2d ->
- convolution 2d -> relu -> max pooling 2d ->
- 2d to 1d vector -> linear -> relu ->
- linear -> relu ->
- linear -> relu ->
- concatenate with input2 -> 
- linear -> relu ->
- linear -> relu ->
- linear -> output
- output -> array(25) # The array of positions where the checker moves  (on 0 index is move from jail)
