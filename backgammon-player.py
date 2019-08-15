from natsort import natsorted
import PIL
from PIL import Image
from PIL import ImageDraw as draw
from PIL import ImageFont as ifont

class Board:
#     strView = ''
    def __init__(self):
        self.myBoard = {}
        for i in range(24):
            self.myBoard[i] = 0
        self.myBoard[0] = 2
        self.myBoard[5] = -5
        self.myBoard[7] = -3
        self.myBoard[11] = 5
        self.myBoard[12] = -5
        self.myBoard[16] = 3
        self.myBoard[18] = 5
        self.myBoard[23] = -2
        self.maxRows = 5
        self.xFree = 0
        self.oFree = 0
        self.xJail = 0
        self.oJail = 0
        self.xHome = 5
        self.oHome = 5
        
        self.mapping_side = {
            False: -1,
            True: 1
        }
        
        self.strView = ''        
        
    def toImage(self, step_i, folder='.'):
        lines = self.__repr__().split('\n')
#         lines = self.strView.split('\n')
        max_count = 0
        #font = ifont.truetype("couriernew.ttf",20)
        font = ifont.truetype("CourierNew-B.ttf", 20)
        lines = lines[2:-2]
        for l in lines:
            max_count = max(len(l), max_count)
        img = Image.new("L", (max_count*14, (len(lines)-1)*20), 255)
        pix = img.load()
        paint = draw.Draw(img)
        for i,l in enumerate(lines):
            paint.text(( 0, 20*i), l, font=font)
        if img.size[0] > img.size[1]:
            img = img.resize((img.size[0],img.size[0]))
        else:
            img = img.resize((img.size[1], img.size[1]))
        img = img.resize((512,512))
        img.save(os.path.join(folder, str(step_i) + ".png"))
        return

    def getPossibleMoves(self, side, allSteps=[]):
        
        allSpaces = list(range(len(self.myBoard)))
        if (side and self.xJail>0) or (not side and self.oJail>0):
            allSpaces.append(-1)    
            
        availableMoves = []
        for steps in allSteps:
            for space in allSpaces:
                if side:                    
                    if self.xJail > 0 and space != -1:                     
                        continue
                    elif self.xJail > 0 and space == -1:
                        if self.myBoard[24-steps] > 1:
                            continue
                        elif self.myBoard[24-steps] == 1:
                            availableMoves.append([space, side, steps, 0.5])
                        else:
                            availableMoves.append([space, side, steps, 0.0])
                    elif space not in self.myBoard:
                        continue
                    elif self.myBoard[space] >= 0:
                        continue
                    else:
                        newSpace = space - steps
                        if newSpace < 0:
                            if self.xHome < 15:
                                continue
                            else:
                                availableMoves.append([space, side, steps, 1.0])
                        elif (self.myBoard[newSpace] > 1):
                            continue
                        elif (self.myBoard[newSpace] == 1):
                            availableMoves.append([space, side, steps, 0.5])
                        else:         
                            availableMoves.append([space, side, steps, 0.1])
        
                else:
                    if self.oJail > 0 and space != -1:
                        continue                    
                    elif self.oJail > 0 and space == -1:
                        if -self.myBoard[steps-1] > 1 :
                            continue
                        elif -self.myBoard[steps-1] == 1:                        
                            availableMoves.append([space, side, steps, 0.5])                            
                        else:
                            availableMoves.append([space, side, steps, 0.0])
                    elif space not in self.myBoard:
                        continue
                    elif -self.myBoard[space] >= 0:
                        continue
                    else:                        
                        newSpace = space + steps
                        if newSpace > 23:
                            if self.oHome < 15:
                                continue
                            else:
                                availableMoves.append([space, side, steps, 1.0])
                        elif -self.myBoard[newSpace] > 1:
                            continue
                        elif -self.myBoard[newSpace] == 1:
                            availableMoves.append([space, side, steps, 0.5])
                        else:
                            availableMoves.append([space, side, steps, 0.1])
                            
        return natsorted(availableMoves)
            

    def makeMove(self, space, side, steps):
        """
        @param
        space = starting space
        side = true if X (steps are decreasing), false if O (steps are increasing)
        steps = number of steps
        *** if you are trying to free a piece from jail, steps = 0
        @return
        tuple of true if move was made, false if not possible, and string with response
        """
        if side:
            
            if self.xJail > 0 and space != -1: 
                return (False, "Make sure you free your piece from jail first!")
            elif self.xJail > 0 and space == -1:
                newSpace = 24 - steps
                if self.myBoard[newSpace] > 1:
                    return (False, "Space is occupied")
                elif self.myBoard[newSpace] == 1:
                    self.xJail -= 1
                    self.myBoard[newSpace] = -1
                    self.oJail += 1
                    self.oHome -= 1
                    return (True, "Piece released from jail and sent one to jail!")
                else:
                    self.xJail -= 1
                    self.myBoard[newSpace] -= 1
                    return (True, "Piece released from jail!")
            elif space not in self.myBoard:
                return (False, "Space is wrong")
            elif (self.myBoard[space] >= 0):
                return (False, "Space is either empty or wrong team")
            else:
                newSpace = space - steps
                if newSpace < 0:
                    if self.xHome < 15:
                        return (False, "Make sure all of your pieces are in home before clearing them!")
                    else:
                        self.myBoard[space] += 1
                        self.xFree += 1
                        return (True, "Piece cleared!")
                elif (self.myBoard[newSpace] > 1):
                    return (False, "New space is occupied")
                elif (self.myBoard[newSpace] == 1):
                    self.myBoard[space] += 1
                    self.myBoard[newSpace] = -1
                    self.oJail += 1
                    if newSpace >= 18:
                        self.oHome -= 1
                    return (True, "Sent one to jail!")
                else:
                    self.myBoard[space] += 1
                    self.myBoard[newSpace] -= 1
                    if newSpace < 6:
                        self.xHome += 1
                    if newSpace < 11:
                        self.updateRows(True)
                    else:
                        self.updateRows(False)
                    return (True, "Move made")
        else:
            if self.oJail > 0 and space != -1:
                return (False, "Make sure you free your piece from jail first!")
            elif self.oJail > 0 and space == -1:
                newSpace = -1+steps
                if -self.myBoard[newSpace] > 1 :
                    return (False, "Space is occupied")
                elif -self.myBoard[newSpace] == 1: 
                    self.oJail -= 1
                    self.myBoard[newSpace] = 1
                    self.xJail += 1
                    self.xHome -= 1
                    return (True, "Piece released from jail and one sent to jail!")
                else:
                    self.oJail -= 1
                    self.myBoard[newSpace] +=  1
                    return (True, "Piece released from jail!")
            elif space not in self.myBoard:                
                return (False, "Space is wrong")
            elif -self.myBoard[space] >= 0:
                return (False, "Space is either empty or wrong team")
            else:
                newSpace = space + steps
                if newSpace > 23:
                    if self.oHome < 15:
                        return (False, "Make sure all of your pieces are in home before clearing them!")
                    else:
                        self.myBoard[space] -= 1
                        self.oFree += 1
                        return (True, "Piece cleared!")
                elif -self.myBoard[newSpace] > 1:
                    return (False, "New space is occupied")
                elif -self.myBoard[newSpace] == 1:
                    self.myBoard[space] -= 1
                    self.myBoard[newSpace] = 1
                    self.xJail += 1
                    if newSpace < 6:
                        self.xHome -= 1
                    return (True, "Sent one to jail!")
                else:
                    self.myBoard[space] -= 1
                    self.myBoard[newSpace] += 1
                    if newSpace > 17:
                        self.oHome += 1
                    if newSpace < 11:
                        self.updateRows(True)
                    else:
                        self.updateRows(False)
                    return (True, "Move made")

    def updateRows(self, top):
        changed = False
        if top:
            for i in range(12):
                if (self.myBoard[i] > 5):
                    self.maxRows = self.myBoard[i]
                    changed = True
        else:
            for i in range(23,11,-1):
                if (self.myBoard[i] > 5):
                    self.maxRows = self.myBoard[i]
                    changed = True
        if not changed:
            self.maxRows = 5


    def __repr__(self):
        """
        1 -> 1
        2 -> 5
        3 -> 9
        4 -> 13
        5 -> 17
        6 -> 21
        7 -> 28
        8 -> 32
        9 -> 36
        10 -> 40
        11 -> 44
        12 -> 48
        13 -> 1
        17 -> 17
        19 -> 28
        24 -> 48
        """

        emptyline = "|                       | |                       |"
#         if (self.oJail > 0):
        boardstring = "	                O Jail: " + str(self.oJail) + "\n"
        boardstring += "                X HOME BOARD     Freed:" + str(self.xFree) + "\n"
#         else:
#             boardstring = "	            X HOME BOARD     Freed:" + str(self.xFree) + "\n"
        boardstring += " -------------------------------------------------\n"
        boardstring += "|12  11  10  9   8   7  | | 6   5   4   3   2   1 |\n"
        for i in range(self.maxRows):
            boardstring += self.populateTop(i)
        boardstring += " ------------------------------------------------- \n"
        for i in range(self.maxRows-1,-1,-1):
            boardstring += self.populateBottom(i)
        boardstring += "|13  14  15  16  17  18 | | 19  20  21  22  23  24|\n"
        boardstring += " -------------------------------------------------\n"
        boardstring += "                O HOME BOARD     Freed: " + str(self.oFree) + "\n"
#         if (self.xJail > 0):
        boardstring += "                    X Jail: " + str(self.xJail) + "\n"
        self.strView = boardstring
        return boardstring


    def populateTop(self, lineNumber):
        """
        1 -> 1
        2 -> 5
        3 -> 9
        4 -> 13
        5 -> 17
        6 -> 21
        7 -> 28
        8 -> 32
        9 -> 36
        10 -> 40
        11 -> 44
        12 -> 48
        """
        line = "|                       | |                       |\n"
        boardtostring = {} 
        boardtostring[0] = 48
        boardtostring[1] = 44
        boardtostring[2] = 40
        boardtostring[3] = 36
        boardtostring[4] = 32
        boardtostring[5] = 28
        boardtostring[6] = 21
        boardtostring[7] = 17
        boardtostring[8] = 13
        boardtostring[9] = 9
        boardtostring[10] = 5
        boardtostring[11] = 1
        for i in range(12):
            if (self.myBoard[i] > lineNumber):
                line = line[:boardtostring[i]] + 'O' + line[boardtostring[i]+1:]
            elif (abs(self.myBoard[i]) > lineNumber and self.myBoard[i] < 0):
                line = line[:boardtostring[i]] + 'X' + line[boardtostring[i]+1:]
        return line

    def populateBottom(self, lineNumber):
        line = "|                       | |                       |\n"
        boardtostring = {}
        boardtostring[11] = 48
        boardtostring[10] = 44
        boardtostring[9] = 40
        boardtostring[8] = 36
        boardtostring[7] = 32
        boardtostring[6] = 28
        boardtostring[5] = 21
        boardtostring[4] = 17
        boardtostring[3] = 13
        boardtostring[2] = 9
        boardtostring[1] = 5
        boardtostring[0] = 1
        for i in range(12):
            if (self.myBoard[12+i] > lineNumber):
                line = line[:boardtostring[i]] + 'O' + line[boardtostring[i]+1:]
            elif (abs(self.myBoard[12+i]) > lineNumber and self.myBoard[12+i] < 0):
                line = line[:boardtostring[i]] + 'X' + line[boardtostring[i]+1:]
        return line
        
        
        
        
        
# from board import Board
import random
import sys

MOVE_TYPES = [
    'JailFree',
    'JailFreeWithJailSend',
    'Move',
    'MoveJailSend',
    'MoveMakeSingle',
    'MoveLeftSingle',
    'MoveRelieve'   
]

exitTerms = ("quit", "exit", "bye","q")

SAVE_GAMES = True

MOVES_HISTORY = []
iMove=0
wins = []

def main(gameNumber=0):
    
    iMove=0
    b = Board()   
    
    intro = open('readme.txt', 'r')

    SIDE = True #True if X, false if O
    for line in intro:
        print(line)
    print("What do you want to play? Type 'pc' for Player vs. Computer or 'pp' for Player vs. Player")

#     line = input()
    line = 'pc'
    if line == 'pc':
        while (line not in exitTerms and (b.xFree < 15 or b.oFree < 15)):
            print(b)
            
            roll1 = random.randint(1,6)
            roll2 = random.randint(1,6)
            turnComplete = False
            
            total = roll1 + roll2
            if roll1 == roll2:
                total *= 2
                
            rolls = [roll1, roll2] if roll1 != roll2 else [roll1, roll2]*2            
            availableMoves = b.getPossibleMoves(SIDE, rolls)
            
            if SIDE:
                print("X, You rolled dice and you have " + '+'.join([str(r) for r in rolls]) + ' steps')
            else:
                print("O, You rolled dice and you have " + '+'.join([str(r) for r in rolls]) + ' steps')
                
            while (not turnComplete and line not in exitTerms and len(rolls) > 0):                
                
                print("Possible Moves:" , [str(m[0]+1)+', '+ str(m[2]) for m in availableMoves])
                
                if SIDE:
#                     line = input()                    
                    if len(availableMoves)>0:
                        chosenMove = random.sample(availableMoves, 1)[0]
                        line = str(chosenMove[0]+1)+', '+ str(chosenMove[2])
                    else:
                        line = 'f'
                    print('Machine chose %s' % line)
                else:
#                     line = input()  
                    if len(availableMoves)>0:
                        chosenMove = random.sample(availableMoves, 1)[0]
                        line = str(chosenMove[0]+1)+', '+ str(chosenMove[2])
                    else:
                        line = 'f'
                    print('Machine chose %s' % line)
                
                try:
                    space,steps = parseInput(line)
                except:
                    print('Wrong Input')
                    continue
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
                    print("You didn't roll that!")
                    continue
                    
                if not jailCase and space==0:
                    print("No checkers in the jail!")
                    # Must jump to beginning of loop                
                
                if space == 0 and SIDE and b.xJail > 0:
                    jailFreed = True
                elif space == 0 and not SIDE and b.oJail > 0:
                    jailFreed = True
                        
                space = space - 1
                if space < -1 or space > 23:# or steps < 0:
                    print("That move is not allowed.  Please try again.")
                    continue
                    #Same deal here.
                    
                # History 
                # board_state = copy.deepcopy(b.my) 
                hist_state = list(b.myBoard.values()) + [b.oJail, b.xJail]
                hist_move = [space, SIDE, steps]
                move, response = b.makeMove(space, SIDE, steps)
                print(response)
                
#                 if move and jailFreed:
#                     steps = tempSteps
                    
                if move:
                    total = total - steps
                    print(b)
                                    
                    MOVES_HISTORY.append({'hist_state':hist_state, 'hist_move':hist_move, 'available_moves':availableMoves})
                    
                    b.toImage(iMove, CURRENT_FOLDER)
                    iMove+=1  
                    
#                     if steps in rolls:
                    rolls.remove(steps)
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
                
            SIDE = not SIDE
    else:
        print('You dont need this for now')
#         while (line not in exitTerms and (b.xFree < 15 or b.oFree < 15)):
#             print(b)
#             roll1 = random.randint(1,6)
#             roll2 = random.randint(1,6)
#             turnComplete = False
#             total = roll1 + roll2
#             if (roll1 == roll2):
#                 total *= 2
#             print("You rolled a " + str(roll1) + " and a " + str(roll2) + " giving you a total of " + str(total) + " moves.")
#             if SIDE:
#                 print("X, what do you want to do?")
#             else:
#                 print("O, what do you want to do?")
#             while (not turnComplete and line not in exitTerms and total > 0):
#                 line = input()
#                 try:
#                     space,steps = parseInput(line)
#                 except:
#                     print('Wrong Input')
#                     continue
#                 jailFreed = False
#                 jailCase = False
#                 if (SIDE and b.xJail > 0):
#                     jailCase = True
#                 if (not SIDE and b.oJail > 0):
#                     jailCase = True
#                 if (space == 100 and steps == 100):
#                     total = 0
#                     break
#                 if (space == 101 and steps == 101):
#                     break
#                 if (steps != roll1 and steps != roll2 and steps != (roll1 + roll2) and steps != 100 and not jailCase):
#                     print("You didn't roll that!")
#                     continue
#                     # Must jump to beginning of loop
#                 space = space - 1
#                 if (steps == 0 and SIDE and b.xJail > 0):
#                     tempSteps = space - 18
#                     if (tempSteps != roll1 and tempSteps != roll2):
#                         print("You didn't roll that!")
#                         continue
# #                     if (steps != roll1 and steps != roll2):
# #                         print("You didn't roll that!")
# #                         continue
#                     else:
#                         jailFreed = True
#                 elif (steps == 0 and not SIDE and b.oJail > 0):
#                     tempSteps = space + 1
#                     if (tempSteps != roll1 and tempSteps != roll2):
#                         print("You didn't roll that!")
#                         continue
#                     else:
#                         jailFreed = True
#                 if (space < 0 or space > 23 or steps < 0):
#                     print("That move is not allowed.  Please try again.")
#                     continue
#                     #Same deal here.                
#                 move, response = b.makeMove(space, SIDE, steps)
#                 print(response)
#                 if (move and jailFreed):
#                     steps = tempSteps
#                 if move:
#                     total = total - steps
#                     print(b)
#                     print("You have " + str(total) + " steps left.")
#             SIDE = not SIDE


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

# if __name__ == "__main__":
#     main()


import os
import pandas as pd

if not os.path.exists( 'data' ):
    os.mkdir('data')

prefix = 'data/game'
wins = []
header = list(range(0,24)) + ['oJail', 'xJail'] + ['ChosenMove', 'Player' ,'ChosenTurnNumber', 'AvailableMoves']

for igame in range(200):
    
    CURRENT_FOLDER = prefix+str(igame)
    
    if not os.path.exists( CURRENT_FOLDER ):
        os.mkdir(CURRENT_FOLDER)
        
    MOVES_HISTORY = []
    iMove=0  
    
    main(igame)    
    
    filename = CURRENT_FOLDER + '/moves.csv'
    df = pd.DataFrame( [i['hist_state']+i['hist_move']+[i['available_moves']] for i in MOVES_HISTORY] , columns = header )
    df.to_csv(filename, index=False)
    
    #if igame>3:
    #    break
        
df_wins=pd.DataFrame( wins )
df_wins.columns = ['game', 'winner']
df_wins.to_csv('data/result.txt',index=False) 