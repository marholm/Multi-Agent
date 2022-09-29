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
import math

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
        return successorGameState.getScore()

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

        #util.raiseNotDefined()
        # Root-level action-selection
        actions = gameState.getLegalActions(0)
        best_action = ''
        score = -10000000000

        # Iterates all legal actions

        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            # Iterates the actions of the ghosts
            # Number of agents minus pacman minus index 0
            for i in range(1, gameState.getNumAgents() - 1):
                min_value = self.min_value(next_state, 0, i)
                # If value generated is larger than current score, value and action is updated
                if min_value > score:
                    score = min_value
                    best_action = next_state
        return best_action

    # Returns the maximum value
    def max_value(self, gameState, depth):
        # Defining variables for depth, the max-value and legal actions for max-agents
        # For every function-call, depth is increased by 1
        current_depth = depth + 1
        agent_actions = gameState.getLegalActions(0)
        max_v = -10000000000

        # Checking if function is in a terminal state
        if self.terminal_state(gameState, current_depth):
            return self.evaluationFunction(gameState)

        #For-loop for choosing best action for the max-agent (pacman)
        for action in agent_actions:
            for i in range(1, gameState.getNumAgents() - 1):
                succ = gameState.generateSuccessor(0, action)
                max_v = max(max_v, self.min_value(succ, current_depth, i))
        return max_v


    # Return the minimum-value
    def min_value(self, gameState, depth, ghost_index):
        # Defining variables for depth, the min-value, legal actions for min-agents
        current_depth = depth + 1
        ghost_actions = gameState.getLegalActions(ghost_index)
        min_v = 10000000000

        # Checking if function is in a terminal state
        if self.terminal_state(gameState, current_depth):
            return self.evaluationFunction(gameState)

        # For-loop that goes through all actions of the ghosts
        for action in ghost_actions:
            succ = gameState.generateSuccessor(ghost_index, action)
            min_v = min(min_v, self.max_value(succ, current_depth))
        return min_v

    #Terminal-state fuction, returns true if agent is in a terminal state
    def terminal_state(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
            return True
        else:
            return False

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        actions = gameState.getLegalActions(0)
        score = -10000000000
        best_action = ''
        alpha = -10000000000
        beta = 10000000000

        # Iterates all legal actions
        for action in actions:
            succ = gameState.generateSuccessor(0, action)

            # Goes through actions of the ghosts
            for i in range(1, gameState.getNumAgents()-1):
                min_value = self.minValue(succ, 1, i, alpha, beta)

                # Update action and score values if value of score is larger than value of the min_value
                if min_value > score:
                    best_action = action
                    score = min_value

                # Return the best action if min_value is larger than beta-value
                if min_value > beta:
                    return best_action

                #Set new alpha value
                alpha = max(alpha, score)
        return best_action

    def maxValue(self, gameState, depth, alpha, beta):
        # Defining variables for depth, alpha, beta, the max-value and legal actions for the max-agent
        current_depth = depth + 1
        current_alpha = alpha
        agent_actions = gameState.getLegalActions(0)
        max_v = -10000000000

        # Checking if the max-agent is in a terminal state
        if self.terminal_state(gameState, current_depth):
            return self.evaluationFunction(gameState)

        # For-loop for choosing best action for the max-agent (pacman)
        for action in agent_actions:
            succ = gameState.generateSuccessor(0, action)

            for i in range(1, gameState.getNumAgents()-1):
                max_v = max(max_v, self.minValue(succ, current_depth, i, current_alpha, beta))
                # If max-value is larger than beta, we return the max value
                if max_v >= beta:
                    return max_v
                # Else, we update or alpha value to the larges of the current alpha and our max-value
                else:
                    current_alpha = max(current_alpha, max_v)
        return max_v

    def minValue(self, gameState, depth, ghost_index, alpha, beta):
        # Defining variables for depth, beta, the min-value and legal actions for min-agents
        current_depth =  depth + 1
        current_beta = beta
        min_v = 10000000000
        ghost_actions = gameState.getLegalActions(ghost_index)

        # Checking if the function is in a terminal state
        if self.terminal_state(gameState, current_depth):
            return self.evaluationFunction(gameState)

        # For-loop for choosing best actions for the min-agents (ghosts)
        for action in ghost_actions:
            succ = gameState.generateSuccessor(ghost_index, action)
            min_v = min(min_v, self.maxValue(succ, current_depth, alpha, current_beta))

            # Checks if the min-value is smaller or equal than the alpha
            if min_v <= alpha:
                return min_v
            # Else, update beta to min of current-beta and min-value
            else:
                current_beta = min(current_beta, min_v)
        return min_v

    # Terminal-state fuction, returns true if agent is in a terminal state
    def terminal_state(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
            return True
        else:
            return False


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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
