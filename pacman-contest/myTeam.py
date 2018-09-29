# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import time

import numpy as np
from game import Actions
import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint

GENERIC = False
BELIEF_LOGIC = True
beliefs = []
beliefsInitialized = []
USE_BELIEF_DISTANCE = True
DEBUG = False
validNextPositions = {}
Walls = set()
NoWalls = set()
nearestEnemyLocation = None
POWER_PELLET_VICINITY = 5
DEFENDING = []
DNUM = 0
nearestEnemyLocation = None
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.isTraining = False
        self.episodesSoFar = 0
        self.epsilon = 0.05
        self.discountFactor = 0.75
        self.alphaLR = 0.0000000002
        self.PrevAction = None
        self.minPelletsToCashIn = 6
        self.maxPelletsToCashIn = 15
        self.AttackHistory = []


    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # Offline data computation, can be utilised further.
        arr = np.zeros((32, 16))
        noWallsTemp = set([(index[0][0] + 1, index[0][1] + 1) for index in np.ndenumerate(arr) if
                           not gameState.hasWall(index[0][0] + 1, index[0][1] + 1)])
        WallsTemp = set([(index[0][0] + 1, index[0][1] + 1) for index in np.ndenumerate(arr) if
                         gameState.hasWall(index[0][0] + 1, index[0][1] + 1)])
        # Valid Moves contains all available moves from each available legal state in the map
        for x, y in noWallsTemp:
            availableMoves = []
            if (x + 1, y) not in WallsTemp and 0 < x + 1 < 33 and 0 < y < 17:
                availableMoves.append((x + 1, y))
            if (x, y + 1) not in WallsTemp and 0 < x < 33 and 0 < y + 1 < 17:
                availableMoves.append((x, y + 1))
            if (x - 1, y) not in WallsTemp and 0 < x - 1 < 33 and 0 < y < 17:
                availableMoves.append((x - 1, y))
            if (x, y - 1) not in WallsTemp and 0 < x < 33 and 0 < y - 1 < 17:
                availableMoves.append((x, y - 1))
            global validNextPositions
            key = str(x) + ',' + str(y)
            validNextPositions[key] = availableMoves
        global Walls
        Walls = WallsTemp
        global NoWalls
        NoWalls = noWallsTemp
        if self.index % 2 == 0:
            startState = gameState.getAgentState(1).getPosition()
        else:
            startState = gameState.getAgentState(0).getPosition()
        bell = util.Counter().fromkeys(NoWalls, 1)
        bell[startState] = 1
        global beliefsOpponent1
        beliefsOpponent1 = bell
        global beliefsOpponent2
        beliefsOpponent2 = bell
        global OpponentLocation1
        OpponentLocation1 = str(startState[0]) + ',' + str(startState[1])
        global OpponentLocation2
        OpponentLocation2 = str(startState[0]) + ',' + str(startState[1])

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        # Append the History of all States played
        self.observationHistory.append(gameState)
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        values = [self.AproaxQvalue(gameState, a) for a in actions]
        #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        bestAction = None
        # Go back to start if there are only 5 food left
        if foodLeft <= 0:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist

        else:
            bestAction = random.choice(bestActions)
        #self.PrevAction = bestAction
        return bestAction

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def AproaxQvalue(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def ValueFromQvalue(self, gameState):

        '''
         Given a state this function caliculates the best Q value for the next state i.e, Q(s',a')
        '''

        actions = gameState.getLegalActions(self.index)

        if actions:
            values = [self.AproaxQvalue(gameState, a) for a in actions]
            maxValue = max(values)
            return maxValue
        else:
            return 0



    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def getReward(self, gameState):
        foodList = self.getFood(gameState).asList()

        '''
        This is reward function which returns the Cummilative reward in form of a rewward shaping.
        '''
        prev_gameState = self.observationHistory.pop()
        # BRING BACK FOOD TO GET MORE REWARD
        # food current pacman is carrying
        prev_food_carrying = prev_gameState.getAgentState(self.index).numCarrying
        food_carrying = gameState.getAgentState(self.index).numCarrying
        prev_deposited = prev_gameState.getAgentState(self.index).numReturned
        food_deposited = gameState.getAgentState(self.index).numReturned
        food_brought_home = food_deposited - prev_deposited

        net_change_food_carried = food_carrying - prev_food_carrying
        # small reward for eating power capusule

        # small reward for eating food
        if net_change_food_carried > 0:
            eat_reward = 0.2
        else:
            eat_reward = 0

        # small reward for eating capusules
        mypellets_prev = len(self.getCapsules(prev_gameState))
        mypellets_now = len(self.getCapsules(gameState))
        netChangePellets = mypellets_prev - mypellets_now
        if netChangePellets > 0:
            IAtePellete = 1
        else:
            IAtePellete = 0

        # if brought food home give value +10
        if food_brought_home > 0:
            bring_food_value = 20 * food_brought_home
        else:
            bring_food_value = 0

        # REWARD FOR EATING ENEMY PACMAN
        myAgentPosition = prev_gameState.getAgentPosition(self.index)
        eat_enemy_value = 0
        enemy_ate_us_value = 0
        for opponent in self.getOpponents(gameState):
            maybePosition = prev_gameState.getAgentPosition(opponent)
            mayBePositionNow = gameState.getAgentPosition(opponent)
            WasIaPacman = prev_gameState.getAgentState(self.index).isPacman
            AmIaPacman = gameState.getAgentState(self.index).isPacman
            # Enemy is a pacman and he was 1 distance away from us in previous game state

            if maybePosition != None:
                IsEnemyPacman = prev_gameState.getAgentState(opponent).isPacman
                howFarwasEnemy = self.getMazeDistance(myAgentPosition, maybePosition)
                howFarisEnemy = self.getMazeDistance(myAgentPosition, mayBePositionNow)
                if IsEnemyPacman:
                    if howFarwasEnemy < 2:
                        # Enemy has disappeared in the current state means we ate him
                        if howFarisEnemy > 10:
                            eat_enemy_value = 100
                # If I was a pacman and I turned into a ghost and returned to begining
                elif WasIaPacman == True and AmIaPacman == False:
                    myAgentCurrenPos = gameState.getAgentPosition(self.index)
                    if self.start == myAgentCurrenPos:
                        enemy_ate_us_value = -100  # negative reward for being eaten

        # NEGATIVE REWARDS
        ourFoodNow = len(self.getFoodYouAreDefending(gameState).asList())
        ourFoodPrev = len(self.getFoodYouAreDefending(prev_gameState).asList())
        netOurFoodChange = ourFoodPrev - ourFoodNow

        # small negative reward if enemy is eating our food

        if netOurFoodChange > 0:
            enemy_eating_value = -0.2
        else:
            enemy_eating_value = 0

        # small negative reward for enemy eating power pelletes

        pellets_prev = len(self.getCapsulesYouAreDefending(prev_gameState))
        pellets_now = len(self.getCapsulesYouAreDefending(gameState))
        netChangePellets = pellets_prev - pellets_now
        if netChangePellets > 0:
            enemyAtePellete = -1
        else:
            enemyAtePellete = 0

        cummilativeReward = (eat_enemy_value + eat_reward + bring_food_value +
                             enemy_eating_value + enemyAtePellete + enemy_ate_us_value + IAtePellete)

        return cummilativeReward

    def observationFunction(self, gameState):

       '''
        Note this observationFuntion ovverides the function in CaptureAgents

       '''
       if len(self.observationHistory) > 0 and self.isTraining:
           self.update(self.observationHistory.pop(), self.lastAction, gameState, self.getReward(gameState))
            # print self.getReward(gameState)

       return gameState.makeObservation(self.index)


    def update(self, state, action, nextState, reward):

        '''

        This update function updates the weights in the Training phase based on every transition:
        Note: We initial the weights to some values we think are good and then learn them with a slow learning
        rate
        '''

        TD = (reward + self.discountFactor * self.ValueFromQvalue(nextState))
        Qvalue =  self.AproaxQvalue(state, action)

        updatedWeights = self.weights.copy()

        FeatureValues = self.getFeatures(state, action)

        for feature in FeatureValues:
            newWeight = updatedWeights[feature] + self.alphaLR * (TD-Qvalue) * FeatureValues[feature]
            updatedWeights[feature] = newWeight
        self.weights = updatedWeights

        # print 'UPDATED WEIGHTS ARE'
        # print self.weights

    ######################
    # BELIEF LOGIC BEGIN #
    ######################
    def getMostLikelyGhostPosition(self, ghostAgentIndex):
        return max(beliefs[ghostAgentIndex])

    def initializeBeliefs(self, gameState):
        beliefs.extend([None for x in range(len(self.getOpponents(gameState)) + len(self.getTeam(gameState)))])
        for opponent in self.getOpponents(gameState):
            self.initializeBelief(opponent, gameState)
        beliefsInitialized.append('done')

    def initializeBelief(self, enemyIndex, gameState):
        belief = util.Counter()
        for gridSpaces in NoWalls:
            belief[gridSpaces] = 1.0
        belief.normalize()
        beliefs[enemyIndex] = belief

    def observeAllOpponents(self, gameState):
        if len(beliefsInitialized):
            for opponent in self.getOpponents(gameState):
                self.observeOneOpponent(gameState, opponent)
        else:
            self.initializeBeliefs(gameState)

    def observeOneOpponent(self, gameState, enemyIndex):
        ourPosition = gameState.getAgentPosition(self.index)
        probabilities = util.Counter()
        maybeIndex = gameState.getAgentPosition(enemyIndex)
        noisyDistance = gameState.getAgentDistances()[enemyIndex]
        if maybeIndex is not None:
            probabilities[maybeIndex] = 1
            beliefs[enemyIndex] = probabilities
            return
        for gridSpaces in NoWalls:
            trueDistance = util.manhattanDistance(gridSpaces, ourPosition)
            modelProb = gameState.getDistanceProb(trueDistance, noisyDistance)
            if modelProb > 0:
                oldProb = beliefs[enemyIndex][gridSpaces]
                probabilities[gridSpaces] = (oldProb + 0.001) * modelProb
            else:
                probabilities[gridSpaces] = 0
        probabilities.normalize()
        beliefs[enemyIndex] = probabilities

    ####################
    # BELIEF LOGIC END #
    ####################

    def getSetOfMaximumValues(self, counterDictionary):
        return [key for key in counterDictionary.keys() if counterDictionary[key] == max(counterDictionary.values())]
    ######################
    # Astar Login Begins #
    ######################

    def isGhost(self, gameState, index):
        """
        Returns true ONLY if we can see the agent and it's definitely a ghost
        """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False
        return not (gameState.isOnRedTeam(index) ^ (position[0] < gameState.getWalls().width / 2))

    def isScared(self, gameState, index):
        """
        Says whether or not the given agent is scared
        """
        isScared = bool(gameState.data.agentStates[index].scaredTimer)
        return isScared

    def isPacman(self, gameState, index):
        """
        Returns true ONLY if we can see the agent and it's definitely a pacman
        """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False
        return not (gameState.isOnRedTeam(index) ^ (position[0] >= gameState.getWalls().width / 2))

    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
        """
        Finds the distance between the agent with the given index and its nearest goalPosition
        """
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        # print width, height
        walls = walls.asList()

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change action vectors to integers so they work correctly with indexing
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        # Values are stored a 3-tuples, (Position, Path, TotalCost)

        currentPosition, currentPath, currentTotal = startPosition, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(lambda entry: entry[2] +  # Total cost so far
                                                             (110)*self.getMazeDistance(startPosition, entry[0]) if
                                                    entry[0] in avoidPositions else 0 +  # Avoid enemy locations
                                        min(self.getMazeDistance(entry[0], endPosition) for endPosition in
                                            goalPositions))

        # Keeps track of visited positions
        visited = set([currentPosition])

        while currentPosition not in goalPositions:

            possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for
                                 vector, action in zip(actionVectors, actions)]
            legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    queue.push((position, currentPath + [action], currentTotal + 1))

            # This shouldn't ever happen...But just in case...
            if len(queue.heap) == 0:
                return None
            else:
                currentPosition, currentPath, currentTotal = queue.pop()

        if returnPosition:
            return currentPath, currentPosition
        else:
            return currentPath

    def getFoodDistance(self, gameState, action):  #
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        nearestDistance = 0
        if len(foodList) > 0:
            nearestDistance = min(
                [self.getMazeDistance(myPos, food) + abs(self.favoredY - food[1]) for food in foodList])
        return nearestDistance


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        self.observeAllOpponents(gameState)
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        # Grab all enemies
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemyPacmen = [a for a in enemies if a.isPacman and a.getPosition() != None]
        nonScaredGhosts = [a for a in enemies if
                           not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
        scaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]

        # Computes distance to enemy non scared ghosts we can see
        dists = []
        for index in self.getOpponents(successor):
            enemy = successor.getAgentState(index)
            if enemy in nonScaredGhosts:
                if USE_BELIEF_DISTANCE:
                     dists.append(self.getMazeDistance(myPos, self.getMostLikelyGhostPosition(index)))
                else:
                    dists.append(self.getMazeDistance(myPos, enemy.getPosition()))
        # Use the smallest distance
        if len(dists) > 0:
            smallestDist = min(dists)
            features['ghostDistance'] = smallestDist
        if action == Directions.STOP: features['stop'] = 1




        return features

    def getWeights(self, gameState, action):
        return { 'ghostDistance': -2, 'stop': -100}

    def attackQvalue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        bestValue = -999999
        bestActions = None
        for action in state.getLegalActions(self.index):
            # For each action, if that action is the best then
            # update bestValue and update bestActions to be
            # a list containing only that action.
            # If the action is tied for best, then add it to
            # the list of actions with the best value.
            value = self.attackQvalue(state, action)
            if (DEBUG):
                print
                "ACTION: " + action + "           QVALUE: " + str(value)
            if value > bestValue:
                bestActions = [action]
                bestValue = value
            elif value == bestValue:
                bestActions.append(action)
        if bestActions == None:
            return Directions.STOP  # If no legal actions return None
        return random.choice(bestActions)  # Else choose one of the best actions randomly




    def chooseAction(self, state):  # Addressing the exploration vs exploitation dilemma!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if len(self.AttackHistory):
            prev_state = self.AttackHistory.pop()

            global nearestEnemyLocation
            if nearestEnemyLocation is None:
                nearestEnemyLocation = state.getInitialAgentPosition(self.getOpponents(state)[0])

            successor = self.getSuccessor(prev_state, self.PrevAction)
            myPos = successor.getAgentState(self.index).getPosition()
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            Ghosts = [agent for agent in enemies if
                      not agent.isPacman and agent.getPosition() is not None and not agent.scaredTimer > 0]

            dists = []
            for index in self.getOpponents(successor):
                enemy = successor.getAgentState(index)
                if enemy in Ghosts:
                    if USE_BELIEF_DISTANCE:
                        # print index, self.getMostLikelyGhostPosition(index)
                        global nearestEnemyLocation
                        nearestEnemyLocation = self.getMostLikelyGhostPosition(index)
                        dists.append(util.manhattanDistance(myPos, self.getMostLikelyGhostPosition(index)) / 10)
                    else:
                        dists.append(self.getMazeDistance(myPos, enemy.getPosition()))



        # currentPos = state.getAgentPosition(2)
        # print type(currentPos)
        # oldPos = (self.AttackHistory.pop()).getAgentPosition(0)
        # #print type(oldPos)
        # if currentPos[0]-oldPos[0]==1
        # Append game state to observation history...
        self.AttackHistory.append(state)
        self.observeAllOpponents(state)



        ###################################TRY TO DO THIS IN INITIAL 15 SECONDS###########################3
        walls = state.getWalls().asList()
        walls = list(set(walls))
        opponentWalls = []
        if self.index % 2 == 0:
            opponentWalls = [w for w in walls if w[0] > 16]  # WARNING: VALUES HARDCODED
        else:
            opponentWalls = [w for w in walls if w[0] < 17]

        ################################################################################################3
        # Pick Action
        ########################################Astar code added
        food = self.getFood(state)
        enemyIndices = self.getOpponents(state)
        enemyGhostLocations = [state.getAgentPosition(i) for i in enemyIndices if
                               self.isGhost(state, i) and not self.isScared(state, i)]


        avoidPositions = set(enemyGhostLocations)

        capsules = self.getCapsules(state)

        attackablePacmen = [state.getAgentPosition(i) for i in enemyIndices if
                            self.isPacman(state, i) and self.isGhost(state, self.index) and not self.isScared(state,
                                                                                                              self.index)]
        scaredGhostLocations = [state.getAgentPosition(i) for i in self.getOpponents(state) if
                                self.isScared(state, i) and self.isGhost(state, i)]
        goalPositions = set(food.asList() + attackablePacmen + capsules + scaredGhostLocations)

        astar_path = self.aStarSearch(state.getAgentPosition(self.index), state, goalPositions, avoidPositions)
        action_astar = astar_path[0]
        # print "astar_action:",action_astar
        ######################################################################################################################

        actionToBeExecuted = None
        legalActions = state.getLegalActions(self.index)
        if action_astar in legalActions:
            actionToBeExecuted = action_astar
        else:
            actionToBeExecuted =  self.computeActionFromQValues(state)
            # print 'CHOICE', actionToBeExecuted

        '''
            action = None
            if len(legalActions):
              if util.flipCoin(self.epsilon) and self.isTraining():#executes when epsilon = 1
                action = random.choice(legalActions)

                else:
               action = self.computeActionFromQValues(state)


            self.lastAction = action
            '''

        ######################################ASTAR
        self.lastAction = action_astar  # used by observationFunction during training phase

        foodLeft = len(self.getFood(state).asList())
        # Prioritize going back to start if we have <= 2 pellets left
        successor = self.getSuccessor(state, action_astar)
        pellets_eaten_by_0 = successor.getAgentState(0).numCarrying
        pellets_eaten_by_2 = successor.getAgentState(2).numCarrying
        pellets_eaten_by_1 = successor.getAgentState(1).numCarrying
        pellets_eaten_by_3 = successor.getAgentState(3).numCarrying
        ##########################################################s

        '''
        successor = self.getSuccessor(state,action)
            if foodLeft <= 2 :
                bestDist = 9999
                for a in legalActions:
                    successor = self.getSuccessor(state, a)
                    pos2 = successor.getAgentPosition(self.index)
                    dist = self.getMazeDistance(self.start, pos2)
                    if dist < bestDist:
                        actionToBeExecuted = a
                        bestDist = dist
        '''
        nowalls = []
        for i in range(1, state.data.layout.height):

            if state.isRed:
                if state.hasWall(state.data.layout.width / 2 -1, i) == False:
                    nowalls.append((state.data.layout.width / 2 -1, i))
            else:
                if state.hasWall(state.data.layout.width / 2, i) == False:
                    nowalls.append((state.data.layout.width / 2, i))




        actionToReturnHome = None
        if True:  # Red team
            if self.minPelletsToCashIn < state.getAgentState(
                    self.index).numCarrying < self.maxPelletsToCashIn or foodLeft <= 2:
                # print "Run to home..."
                # print "min:", self.minPelletsToCashIn, "mypellets:", (
                #         pellets_eaten_by_0 + pellets_eaten_by_2), "max:", self.maxPelletsToCashIn
                # print "nowalls in the midway", nowalls
                goalPositionsOnWayToHome = set(nowalls)
                avoidPositionsOnWayToHome = set(capsules + scaredGhostLocations + enemyGhostLocations)
                self.getSuccessor(state, action_astar)
                pathToReturnHome = self.aStarSearch(state.getAgentPosition(self.index), state, goalPositionsOnWayToHome,
                                                      avoidPositionsOnWayToHome)

                # print "agentpos", state.getAgentPosition(self.index)
                # print "pathToReturnHome:", pathToReturnHome
                actionToReturnHome = pathToReturnHome[0]
                # print "actionToReturnHome:", actionToReturnHome
                reverseLegalActions = [Directions.REVERSE[i] for i in legalActions]
                # print "actionToReturnHome:",actionToReturnHome
                # print "reverseLegalActions:",reverseLegalActions
                if actionToReturnHome in reverseLegalActions:
                    # print "astar reverse action"
                    actionToBeExecuted = actionToReturnHome
                    # print "actionToReturnHome from IF:", actionToReturnHome
                else:
                    # print "random reverse action"

                    actionToBeExecuted = self.computeActionFromQValues(state)
                    # print 'CHOICE', actionToBeExecuted

                # print "------------------------------------------------------------------------------------"
        if (DEBUG):
            print "AGENT " + str(self.index) + " chose action " + action + "!"



        self.PrevAction = actionToBeExecuted


        return actionToBeExecuted


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState1, action):
        gameState = gameState1.deepCopy()

        # Initializing Beliefs
        features = util.Counter()

        # Our position's successor based on current action:
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemyPacmen = [agent for agent in enemies if agent.isPacman and agent.getPosition() is not None]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
        foodDefend = self.getFoodYouAreDefending(gameState).asList()
        pelletsYouaredefending = self.getCapsulesYouAreDefending(successor)

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        invaderDistance = []
        if features['invaderDistance'] == 0:
            for opp in self.getOpponents(successor):
                if successor.getAgentState(opp).isPacman:
                    invaderDistance.append(util.manhattanDistance(myPos, nearestEnemyLocation) / 1000.0)
        if len(invaderDistance) > 0:
            features['invaderDistance'] = min(invaderDistance)

        if action == Directions.STOP: features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        features['eatingRate'] = DNUM
        if successor.getAgentState(self.getOpponents(gameState)[0]).isPacman or successor.getAgentState(
                self.getOpponents(gameState)[1]).isPacman:
            foodMissing = list(set(DEFENDING).difference(foodDefend))
            if len(foodMissing) == 1:
                foodMissing = foodMissing[0]
                # print foodMissing
                features['eatingRate'] = self.getMazeDistance(myPos, foodMissing)
                global DEFENDING
                DEFENDING = foodDefend
                global DNUM
                DNUM = features['eatingRate']
        #print features['eatingRate']
        if len(foodDefend) > 0:  # This should always be True, but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodDefend])
            features['distanceToFood'] = minDistance

        if len(pelletsYouaredefending) > 0:
            minDistance = min([self.getMazeDistance(myPos, pelet) for pelet in pelletsYouaredefending])
            features['distanceTopower'] = minDistance
        return features


    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -100, 'stop': -100, 'reverse': -2,
                'eatingRate': 500, 'distanceToFood': -0.1, 'distanceTopower': -0.2}
