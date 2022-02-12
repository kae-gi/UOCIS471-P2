# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0
        if successorGameState.isWin():
            return 99999 # pacman won

        # Using manhattanDistance, since it is available!
        # keep track of distance from food
        foodDistance = []
        for food in newFood.asList():
            foodDistance.append(util.manhattanDistance(newPos, food))
        # distance from ghost
        ghostDistance = util.manhattanDistance(newPos, newGhostStates[0].getPosition())

        # check if pacman and ghost position is same
        for ghostState in newGhostStates:
            if ghostState.scaredTimer > 0: # power pellet was ate
                if ghostDistance == 0:
                    score += 200 # reward eating a ghost after power pellet
                score += 50 # reward eating a power pellet
            elif ghostState.scaredTimer == 0: # power pellet was not ate
                # better score if further away from ghost
                if ghostDistance > 0:
                    # give more weight to closest food, more weight to further ghosts
                    score += float(ghostDistance/min(foodDistance))
                # pacman loses
                elif ghostDistance == 0:
                    return -99999 # ran into ghost

        # discourage stopping
        if action == 'Stop':
            score -= 100

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def value(gameState):
            """ Based off function minimax-decision in algorithm from our textbook, 3rd edition. """
            # get the maximized root utility value, maxV, and return it
            maxV = float("-inf")
            bestAction = Directions.STOP
            for action in gameState.getLegalActions(0):
                v = minValue(gameState.generateSuccessor(0, action), 1, 1)
                if v > maxV:
                    maxV = v
                    bestAction = action
            return bestAction

        def maxValue(gameState, depth):
            """ Based off function max-value in algorithm from our textbook, 3rd edition. """
            # terminal test
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            # get utility value, v, and return the maximized utility value, maxV
            maxV = float("-inf")
            for action in gameState.getLegalActions(0):
                v = minValue(gameState.generateSuccessor(0, action), depth + 1, 1)
                if v > maxV:
                    maxV = v
            return maxV

        def minValue(gameState, depth, agentIndex):
            """ Based off function min-value in algorithm from our textbook, 3rd edition. """
            # terminal test
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # get utility value, v, and return the minimized utility value, minV
            minV = float("inf")
            # pacman index
            if agentIndex == gameState.getNumAgents() - 1:
                for action in gameState.getLegalActions(agentIndex):
                    v = maxValue(gameState.generateSuccessor(agentIndex, action), depth)
                    if v < minV:
                        minV = v
                return minV
            # ghost index
            elif agentIndex >= 1:
                for action in gameState.getLegalActions(agentIndex):
                    v = minValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
                    if v < minV:
                        minV = v
                return minV

        return value(gameState)
        util.raiseNotDefined()




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(gameState):
            """
            Based off function alpha-beta-search in algorithm from our textbook, 3rd edition.
            Also based off algorithm given in project page.
            """
            maxV = float("-inf")
            alpha =  float("-inf")
            beta =  float("inf")
            bestAction = Directions.STOP
            for action in gameState.getLegalActions(0):
                v = minValue(gameState.generateSuccessor(0, action), 1, 1, alpha, beta)
                if v > maxV:
                    maxV = v
                    bestAction = action
                if v > beta:
                    return action
                alpha = max(v, alpha)
            return bestAction

        def maxValue(gameState, depth, alpha, beta):
            """
            Based off function max-value in algorithm from our textbook, 3rd edition.
            Also based off algorithm given in project page.
            """
            # terminal test
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            # get utility value, v, and return the maximized utility value, maxV
            maxV = float("-inf")
            for action in gameState.getLegalActions(0):
                v = minValue(gameState.generateSuccessor(0, action), depth + 1, 1, alpha, beta)
                if v > maxV:
                    maxV = v
                if maxV > beta:
                    return maxV
                alpha = max(alpha, maxV)
            return maxV

        def minValue(gameState, depth, agentIndex, alpha, beta):
            """
            Based off function min-value in algorithm from our textbook, 3rd edition.
            Also based off algorithm given in project page.
            """
            # terminal test
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # get utility value, v, and return the minimized utility value, minV
            minV = float("inf")
            # pacman index
            if agentIndex == gameState.getNumAgents() - 1:
                for action in gameState.getLegalActions(agentIndex):
                    v = maxValue(gameState.generateSuccessor(agentIndex, action), depth, alpha, beta)
                    if v < minV:
                        minV = v
                    if minV < alpha:
                        return minV
                    beta = min(beta, minV)
                return minV
            # ghost index
            elif agentIndex >= 1:
                for action in gameState.getLegalActions(agentIndex):
                    v = minValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta)
                    if v < minV:
                        minV = v
                    if minV < alpha:
                        return minV
                    beta = min(beta, minV)
                return minV

        return value(gameState)
        util.raiseNotDefined()




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(gameState):
            """
            Based off function minimax-decision in minimax algorithm from our textbook, 3rd edition.
            Also based off pseudocode presented in lecture 7.
            """
            # get the maximized root utility value, maxV, and return it
            maxV = float("-inf")
            bestAction = Directions.STOP
            for action in gameState.getLegalActions(0):
                v = expValue(gameState.generateSuccessor(0, action), 1, 1)
                if v > maxV:
                    maxV = v
                    bestAction = action
            return bestAction

        def maxValue(gameState, depth):
            """
            Based off function max-value in algorithm from our textbook, 3rd edition.
            Also based off pseudocode presented in lecture 7.
            """
            # terminal test
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            # get utility value, v, and return the maximized utility value, maxV
            maxV = float("-inf")
            for action in gameState.getLegalActions(0):
                v = expValue(gameState.generateSuccessor(0, action), depth + 1, 1)
                if v > maxV:
                    maxV = v
            return maxV

        def expValue(gameState, depth, agentIndex):
            """
            Based off function modified min-value (for expectimax) in algorithm from our textbook, 3rd edition.
            Also based off pseudocode presented in lecture 7.
            """
            # terminal test
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # get utility value, v, and return the expectimax utility value, expV
            expV = 0
            # pacman index
            if agentIndex == gameState.getNumAgents() - 1:
                for action in gameState.getLegalActions(agentIndex):
                    v = maxValue(gameState.generateSuccessor(agentIndex, action), depth)
                    expV += v * float(1/len(gameState.getLegalActions(agentIndex))) # v * probability
                return expV
            # ghost index
            elif agentIndex >= 1:
                for action in gameState.getLegalActions(agentIndex):
                    v = expValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
                    expV += v * float(1/len(gameState.getLegalActions(agentIndex))) # v * probability
                return expV
        return value(gameState)

        util.raiseNotDefined()




def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I used the code from Q1 in the reflex agent, but made some adjustments
    to avoid using actions. I also added in rewarding if pacman was closer to power
    pellets (capsules).

    This function rewards closer distances to foods, and also rewards being farther
    away from ghosts when the scaredTimer is inactive. To promote eating power pellets,
    it rewards being closer to them. If the scaredTimer > 0, it is assumed that a
    power pellet was eaten; this is rewarded, as is being closer to ghosts or eating them
    while the scaredTimer is active.

    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    currCapsule = currentGameState.getCapsules()
    "*** YOUR CODE HERE ***"
    score = 0
    if currentGameState.isWin():
        return 99999 # pacman won

    # Using manhattanDistance, since it is available!
    # keep track of distance from food
    foodDistance = []
    for food in currFood.asList():
        foodDistance.append(util.manhattanDistance(currPos, food))
    # distance from ghost
    ghostDistance = util.manhattanDistance(currPos, currGhostStates[0].getPosition())

    # reward closer to power pellets
    capsuleDistance = []
    if currCapsule:
        for capsule in currCapsule:
            capsuleDistance.append(util.manhattanDistance(currPos, capsule))
            score += float(1/min(capsuleDistance)) # give more weight to closer power pellet

    # check if pacman and ghost position is same
    for ghostState in currGhostStates:
        if ghostState.scaredTimer > 0: # power pellet was ate
            if ghostDistance == 0:
                score += 500 # reward eating a ghost after power pellet
            elif ghostDistance > 0:
                score += float(1/ghostDistance) # reward being closer to ghosts after power pellet
            score += 50 # reward eating a power pellet
        elif ghostState.scaredTimer == 0: # power pellet was not ate
            # better score if further away from ghost
            if ghostDistance > 0:
                # give more weight to closest food, more weight to further ghosts
                score += float(ghostDistance/min(foodDistance))
            # pacman loses
            elif ghostDistance == 0:
                return -99999 # ran into ghost

    return currentGameState.getScore() + score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
