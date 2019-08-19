# Imports

import os, sys
import random
import time
import numpy as np

from natsort import natsorted
from collections import Counter

import pickle
from model import CNN
import torch

import PIL
from PIL import Image
from PIL import ImageDraw as draw
from PIL import ImageFont as ifont
import ImageGenerator as IG

from IPython.display import clear_output

from board import Board


# Load models

# MODEL_CNN_PATH = os.path.join('models','model_n0.pt')
MODEL_CNN_PATH = os.path.join('models','model_n1.pt')
MODEL_QLEARN_PATH = os.path.join('models','model_q1.pkl')

IG.create_alfabet()
cnn = CNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn.load_state_dict(torch.load(MODEL_CNN_PATH))
if device != "cpu":
    cnn.cuda(0)    
    
if os.path.exists(MODEL_QLEARN_PATH):
    with open(MODEL_QLEARN_PATH, 'rb') as f:
        qlearn = pickle.load(f)
else:
    qlearn = {}
    
    
# Game functions
    
exitTerms = ("quit", "exit", "bye","q")

def main(gameNumber=0, VERBOSE = True, SAVE_IMG = False, GAME_TYPE=None, QLEARN = False):    
    
    GAME_TYPES = ['pvp', 'pvr', 'pvq', 'pvd', 'pvc', 'rvr', 'qvr', 'dvr', 'cvr']
    PLAYER_TYPES = {
        'pvp':{True: 'person', False:'person'}, 
        'pvr':{True: 'person', False:'random'}, 
        'pvq':{True: 'person', False:'qlearn'}, 
        'pvd':{True: 'person', False:'dlearn'},
        'pvc':{True: 'person', False:'cnn'},
        'rvr':{True: 'random', False:'random'}, 
        'qvr':{True: 'qlearn', False:'random'},
        'dvr':{True: 'dlearn', False:'random'},
        'cvr':{True: 'cnn',    False:'random'}
    }
    
    b = Board()   
    
    if VERBOSE:    
        with open('readme.txt', 'r') as intro:  
            for line in intro:        
                print(line.replace('\n',''))                
                
    if GAME_TYPE == None:        
        print('What do you want to play? Available game types: {}'.format(GAME_TYPES))
        GAME_TYPE = input() 
        
    line = GAME_TYPE
        
    if GAME_TYPE not in GAME_TYPES:
        print('This game type is not available')
        
    else:
        player_X = PLAYER_TYPES[GAME_TYPE][True]
        player_O = PLAYER_TYPES[GAME_TYPE][False]
        
        iMove = 0        
        SIDE = True #True if X, false if O
        
        while (line not in exitTerms and (b.xFree < 15 or b.oFree < 15)):            
            
            if VERBOSE:
                clear_output()
                print(b)    
            
            roll1 = random.randint(1,6)
            roll2 = random.randint(1,6)
            turnComplete = False
            
            total = roll1 + roll2
            if roll1 == roll2:
                total *= 2
                
            rolls = [roll1, roll2] if roll1 != roll2 else [roll1, roll2]*2            
            availableMoves = b.getPossibleMoves(SIDE, rolls) 
            
            if VERBOSE:
                if SIDE:                
                    print("X, You rolled dice and you have " + '+'.join([str(r) for r in rolls]) + ' steps')
                else:
                    print("O, You rolled dice and you have " + '+'.join([str(r) for r in rolls]) + ' steps')
                
            while (not turnComplete and line not in exitTerms and len(rolls) > 0):                
                
                if VERBOSE:
                    print("Possible Moves:" , [str(m[0]+1)+', '+ str(m[2]) for m in availableMoves])
                
                if SIDE:
                    if player_X == 'person':
                        line = input()
                    elif player_X in ['random', 'qlearn', 'dlearn']:
                        if player_X == 'qlearn':
                            availableMoves = b.filter_best_moves(availableMoves)                            
                        elif player_X == 'dlearn':
                            hist_state_str = str(list(b.myBoard.values()) + [b.oJail, b.xJail])                            

                            if hist_state_str in qlearn:
                                if SIDE in qlearn[hist_state_str]:
                                    
                                    for iam,availMove in enumerate(availableMoves):
#                                         availableMoves[iam][3] = 0.0
                                        if availMove[2] in qlearn[hist_state_str][SIDE]:
                                            if availMove[0] in qlearn[hist_state_str][SIDE][availMove[2]]:
                                                score = qlearn[hist_state_str][SIDE][availMove[2]][availMove[0]] 
                                                availableMoves[iam][3] += score   
                        
                            availableMoves = b.filter_best_moves(availableMoves)   
                            
                        if len(availableMoves) > 0:
                            chosenMove = random.sample(availableMoves, 1)[0]
                            line = str(chosenMove[0]+1)+', '+ str(chosenMove[2])
                        else:
                            line = 'f'
                    elif player_X == 'cnn':                        
                        if len(availableMoves) > 0:
                            _steps = [[m[0],m[2]] for m in availableMoves]

                            np_img = np.ndarray(shape=(1,1,256,128))
                            np_img[0,0] = np.array( b.toImage(step_i=0, save=False).resize((256,128))).transpose((1,0))
                            _image = torch.tensor(np_img).float().to(device)

                            res_tensors = []
                            _steps = np.array(_steps)
                            dice_val = np.unique(_steps[:,1])
                            for step in dice_val:
                                _step = torch.tensor([[step]]).float().to(device)
                                res_tensors.append( cnn(_image, _step) )

                            avalible_pos = []
                            _dict = {}
                            for dice in dice_val:
                                _dict[dice] = []   

                            for s in _steps:
                                _dict[s[1]].append( 
                                    [ res_tensors[ np.where(dice_val == s[1])[0][0] ][0][s[0]].item() , s[0]] 
                                )
                            _dict
                            best_score = -10000.0
                            pair = (0,0)
                            for key in _dict:
                                value = _dict[key]
                                for _v in value:
                                    if best_score < _v[0]:
                                        best_score = _v[0]
                                        pair = (_v[1], key)
                            
                            line = str(pair[0]+1)+', '+ str(pair[1])   
                            
                            # code for old version CNN                            
#                             _steps = [[m[0],m[2]] for m in availableMoves]

#                             np_img = np.ndarray(shape=(1,1,256,128))
#                             np_img[0,0] = np.array( b.toImage(step_i=0, save=False).resize((256,128))).transpose((1,0))
#                             _image = torch.tensor(np_img).float().to(device)

#                             res_tesnors = []
#                             for step in _steps:
#                                 step = torch.tensor([step]).float().to(device)
#                                 res_tesnors.append( cnn(_image, step) )

#                             availableMoves = [[m[0],m[1],m[2],score.item()] for m, score in zip(availableMoves, res_tesnors)]
#                             availableMoves = b.filter_best_moves(availableMoves)    
                            
#                             chosenMove = random.sample(availableMoves, 1)[0]
#                             line = str(chosenMove[0]+1)+', '+ str(chosenMove[2]) 

                        else:
                            line = 'f'
                    else:
                        line = input()                        
                else:
                    if player_O == 'person':
                        line = input()
                    elif player_O in ['random', 'qlearn', 'dlearn']:
                        if player_O == 'qlearn':
                            availableMoves = b.filter_best_moves(availableMoves)   
                        elif player_O == 'dlearn':
                            hist_state_str = str(list(b.myBoard.values()) + [b.oJail, b.xJail])                            

                            if hist_state_str in qlearn:
                                if SIDE in qlearn[hist_state_str]:
                                    
                                    for iam,availMove in enumerate(availableMoves):
                                        if availMove[2] in qlearn[hist_state_str][SIDE]:
                                            if availMove[0] in qlearn[hist_state_str][SIDE][availMove[2]]:
                                                score = qlearn[hist_state_str][SIDE][availMove[2]][availMove[0]] 
                                                availableMoves[iam][3] += score   
                        
                            availableMoves = b.filter_best_moves(availableMoves)
                        if len(availableMoves)>0:
                            chosenMove = random.sample(availableMoves, 1)[0]
                            line = str(chosenMove[0]+1)+', '+ str(chosenMove[2])
                        else:
                            line = 'f'
                            
                    elif player_O == 'cnn':                        
                        if len(availableMoves) > 0:
                            _steps = [[m[0],m[2]] for m in availableMoves]
                                
                            np_img = np.ndarray(shape=(1,1,256,128))
                            np_img[0,0] = np.array( b.toImage(step_i=0, save=False).resize((256,128))).transpose((1,0))
                            _image = torch.tensor(np_img).float().to(device)

                            res_tensors = []
                            _steps = np.array(_steps)
                            dice_val = np.unique(_steps[:,1])
                            for step in dice_val:
                                _step = torch.tensor([[step]]).float().to(device)
                                res_tensors.append( cnn(_image, _step) )

                            avalible_pos = []
                            _dict = {}
                            for dice in dice_val:
                                _dict[dice] = []   

                            for s in _steps:
                                _dict[s[1]].append( 
                                    [ res_tensors[ np.where(dice_val == s[1])[0][0] ][0][s[0]].item() , s[0]] 
                                )
                            best_score = -10000.0
                            pair = (0,0)
                            for key in _dict:
                                value = _dict[key]
                                for _v in value:
                                    if best_score < _v[0]:
                                        best_score = _v[0]
                                        pair = (_v[1], key)
                            
                            line = str(pair[0]+1)+', '+ str(pair[1])     
                           
                        else:
                            line = 'f'
                    else:
                        line = input()
                        
                if VERBOSE:
                    print('Сhoise %s' % line)
                else:
                    print('{}: {}'.format(SIDE, line))
                
                try:
                    space,steps = parseInput(line)
                except:
                    if VERBOSE:
                        print('Wrong Input')
                    continue
                
                # Game logic andrules compliance check
                jailFreed, jailCase = False, False
                
                if (SIDE and b.xJail > 0) or (not SIDE and b.oJail > 0):
                    jailCase = True
                    
                if space == 100 and steps == 100:
                    total = 0
                    rolls = []
                    break
                if space == 101 and steps == 101:
                    break
                if steps not in rolls and steps != 100 and not jailCase:
                    if VERBOSE:
                        print("You didn't roll that!")
                    continue
                    
                if not jailCase and space==0:
                    if VERBOSE:
                        print("No checkers in the jail!")             
                
                if space == 0 and SIDE and b.xJail > 0:
                    jailFreed = True
                elif space == 0 and not SIDE and b.oJail > 0:
                    jailFreed = True
                        
                space = space - 1
                if space < -1 or space > 23:# or steps < 0:
                    if VERBOSE:
                        print("That move is not allowed.  Please try again.")
                    continue
                    
                # History 
                # board_state = copy.deepcopy(b.my) 
                hist_state = list(b.myBoard.values()) + [b.oJail, b.xJail]
                hist_move = [space, SIDE, steps]
                move, response = b.makeMove(space, SIDE, steps)
                if VERBOSE:
                    print(response)
                
#                 if move and jailFreed:
#                     steps = tempSteps
                    
                if move:
                    total = total - steps
                    if VERBOSE:
                        time.sleep(0.05)
                        clear_output()
                        print(b)
                                    
                    MOVES_HISTORY.append({'hist_state':hist_state, 'hist_move':hist_move, 'available_moves':availableMoves})
                                            
                    if SAVE_IMG:
                        b.toImage(iMove, CURRENT_FOLDER, save=True)
                    iMove+=1  
                    
#                     if steps in rolls:
                    rolls.remove(steps)
                    if VERBOSE:
                        print("You have " + str(total) + ' (' + '+'.join([str(r) for r in rolls]) + ") steps left.")  
                    
                availableMoves = b.getPossibleMoves(SIDE, rolls)
            
            if b.oFree>=15:
                print('O wins!')
                wins.append(['game'+str(gameNumber), SIDE])
                break
            elif b.xFree>=15:
                print('X wins!')
                wins.append(['game'+str(gameNumber), SIDE])
                break
                
            if len( b.getPossibleMoves(True, [1,2,3,4,5,6]) )==0 and len( b.getPossibleMoves(False, [1,2,3,4,5,6]) )==0:
                if VERBOSE:
                    print('No available moves!')
                break
                
            SIDE = not SIDE    


#TODO: Include error management
def parseInput(response):
    if response == "d" or response == "f" or response == "done" or response == "finish":
        return(100,100)
    if response in exitTerms:
        return (101, 101)
    # if type(response) == type("Sample string"):
    # 	return(101,101)
    loc = findSeparation(response)
    return(int(response[:loc]), int(response[loc+1:])) 

def findSeparation(value):
    for i in range(len(value)):
        if (value[i] == ' ' or value[i] == ','):
            return i
    return 0
    
    
wins = []
MOVES_HISTORY = [] 
WINNER = None

if __name__ == "__main__":
    main(gameNumber=0, VERBOSE = True, SAVE_IMG = False, GAME_TYPE=None)