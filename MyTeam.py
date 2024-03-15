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
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='CombinedOffensiveAgent', second='DefensiveReflexAgent_avg', num_training=0):
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
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

'''
class AdvancedOffensiveAgent(ReflexCaptureAgent):
    """
    An advanced offensive agent that not only seeks food but also
    attempts to avoid ghosts and considers power pellets.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        # Compute distance to the nearest food
        my_pos = successor.get_agent_state(self.index).get_position()
        if food_list:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Consider ghosts and power pellets
        ghosts = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts_pos = [ghost.get_position() for ghost in ghosts if ghost.get_position() is not None and not ghost.is_pacman]
        if ghosts_pos:
            features['ghost_distance'] = min([self.get_maze_distance(my_pos, ghost_pos) for ghost_pos in ghosts_pos])
        else:
            features['ghost_distance'] = 0

        capsules = self.get_capsules(successor)
        if capsules:
            min_capsule_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])
            features['capsule_distance'] = min_capsule_distance
        else:
            features['capsule_distance'] = 0

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'ghost_distance': 2, 'capsule_distance': -20}'''
      
class StrategicOffensiveAgent(ReflexCaptureAgent):
    """
    An offensive agent that collects food and returns to its own territory to deposit
    points after collecting a certain number of food pellets.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_collected = 0
        self.return_to_home_threshold = 5  # Threshold of food to collect before returning to home territory

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)
        self.home_entry_points = self.get_home_entry_points(game_state)

    def get_home_entry_points(self, game_state):
        # Calculate entry points into home territory
        map_width = game_state.get_walls().width
        mid_x = map_width // 2

        if self.red:  # If agent is on the red team
            entry_x = mid_x - 1  # Entry point just before the midpoint for the red team
        else:
            entry_x = mid_x + 1  # Entry point just after the midpoint for the blue team

        # Finding the open spaces at entry_x
        map_height = game_state.get_walls().height
        entry_points = [(entry_x, y) for y in range(1, map_height) if not game_state.has_wall(entry_x, y)]
        return entry_points

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        my_pos = successor.get_agent_state(self.index).get_position()

        # Update food collected
        if len(self.get_food(game_state).as_list()) > len(food_list):
            self.food_collected += 1

        # Compute distance to the nearest food
        if food_list and self.food_collected < self.return_to_home_threshold:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        else:
            features['distance_to_food'] = 0

        # Go back to home territory if threshold is reached
        if self.food_collected >= self.return_to_home_threshold:
            min_dist_to_home = min([self.get_maze_distance(my_pos, point) for point in self.home_entry_points])
            features['distance_to_home'] = min_dist_to_home
        else:
            features['distance_to_home'] = 0

        # Consider ghosts and power pellets (same as before)
        # [Existing code here for ghost_distance and capsule_distance]

        return features

    def get_weights(self, game_state, action):
        # Adjust the weights based on the agent's current state
        if self.food_collected >= self.return_to_home_threshold:
            return {'successor_score': 100, 'distance_to_food': 0, 'ghost_distance': 2, 'capsule_distance': -20, 'distance_to_home': -10}
        else:
            return {'successor_score': 100, 'distance_to_food': -1, 'ghost_distance': 2, 'capsule_distance': -20, 'distance_to_home': 0}

    def choose_action(self, game_state):
        # Reset food_collected when in home territory
        current_pos = game_state.get_agent_position(self.index)
        if current_pos in self.home_entry_points:
            self.food_collected = 0
        return super().choose_action(game_state)
      
class CombinedOffensiveAgent(ReflexCaptureAgent):
    """
    An offensive agent that seeks food, avoids ghosts, and returns to its own territory 
    to deposit points after collecting a certain number of food pellets.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_collected = 0
        self.return_to_home_threshold = 5  # Threshold of food to collect before returning to home territory
        self.ghost_distance_threshold = 5  # Safe distance from ghosts

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)
        self.home_entry_points = self.get_home_entry_points(game_state)

    def get_home_entry_points(self, game_state):
        # Calculate entry points into home territory
        map_width = game_state.get_walls().width
        mid_x = map_width // 2

        if self.red:  # If agent is on the red team
            entry_x = mid_x - 1  # Entry point just before the midpoint for the red team
        else:
            entry_x = mid_x + 1  # Entry point just after the midpoint for the blue team

        # Finding the open spaces at entry_x
        map_height = game_state.get_walls().height
        entry_points = [(entry_x, y) for y in range(1, map_height) if not game_state.has_wall(entry_x, y)]
        return entry_points

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        my_pos = successor.get_agent_state(self.index).get_position()

        # Update food collected
        if len(self.get_food(game_state).as_list()) > len(food_list):
            self.food_collected += 1

        # Compute distance to the nearest food
        if food_list and self.food_collected < self.return_to_home_threshold:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        else:
            features['distance_to_food'] = 0

        # Avoid ghosts
        ghosts = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts_pos = [ghost.get_position() for ghost in ghosts if ghost.get_position() is not None and not ghost.is_pacman]
        if ghosts_pos:
            min_ghost_distance = min([self.get_maze_distance(my_pos, ghost_pos) for ghost_pos in ghosts_pos])
            if min_ghost_distance < self.ghost_distance_threshold:
                features['ghost_distance'] = min_ghost_distance
        else:
            features['ghost_distance'] = 0

        # Go back to home territory if threshold is reached
        if self.food_collected >= self.return_to_home_threshold:
            min_dist_to_home = min([self.get_maze_distance(my_pos, point) for point in self.home_entry_points])
            features['distance_to_home'] = min_dist_to_home
        else:
            features['distance_to_home'] = 0

        return features

    def get_weights(self, game_state, action):
        # Adjust the weights based on the agent's current state
        if self.food_collected >= self.return_to_home_threshold:
            return {'successor_score': 100, 'distance_to_food': 0, 'ghost_distance': -10, 'distance_to_home': -10}
        else:
            return {'successor_score': 100, 'distance_to_food': -1, 'ghost_distance': -10, 'distance_to_home': 0}

    def choose_action(self, game_state):
        # Reset food_collected when in home territory
        current_pos = game_state.get_agent_position(self.index)
        if current_pos in self.home_entry_points:
            self.food_collected = 0
        return super().choose_action(game_state)


    

from captureAgents import CaptureAgent
import util
from game import Directions
import game
import random

class FoodClusterAgent(CaptureAgent):
    """
    A Pac-Man agent that prioritizes paths with higher densities of food.
    """

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        # Initialization code can go here

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        best_action = None
        best_score = -float('inf')

        for action in actions:
            successor = self.get_successor(game_state, action)
            score = self.evaluate(successor, action)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def get_successor(self, game_state, action):
        return game_state.generate_successor(self.index, action)

    def evaluate(self, successor, action):
        food_list = self.get_food(successor).as_list()
        my_pos = successor.get_agent_state(self.index).get_position()

        food_clusters = self.cluster_food(food_list, my_pos)
        best_cluster = self.find_best_cluster(food_clusters, my_pos)

        if best_cluster:
            distance = self.get_maze_distance(my_pos, best_cluster)
            return -distance  # Negative because smaller distances are better
        return self.get_score(successor)

    def cluster_food(self, food_list, my_pos):
        # Implement clustering algorithm here
        # This is a simple placeholder implementation
        return food_list

    def find_best_cluster(self, food_clusters, my_pos):
        # Find the cluster with the highest food density
        best_cluster = None
        best_density = 0

        for cluster in food_clusters:
            path_length = self.get_maze_distance(my_pos, cluster)  # Simplified
            if path_length > 0:
                density = len(cluster) / float(path_length)
                if density > best_density:
                    best_density = density
                    best_cluster = cluster

        return best_cluster


############## Defensive Agents ##############

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}



class DefensiveReflexAgent_avg(ReflexCaptureAgent):
    """
    An enhanced reflex agent that keeps its side Pacman-free. 
    Improvements include better handling of invader distances.
    
    
    """

    def __init__(self, *args, **kwargs):
        super(DefensiveReflexAgent_avg, self).__init__(*args, **kwargs)
        self.last_move = None
        self.patrol_index = 0  # Initialize patrol index


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            features['average_invader_distance'] = sum(dists) / len(dists)
        else:
            features['invader_distance'] = 0
            features['average_invader_distance'] = 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        # Penalize reverse of the last move
        if self.last_move is not None and Directions.REVERSE[self.last_move] == action:
          features['reverse_last_move'] = 1
        # Patrol feature
        features['patrol'] = 0
        if len(invaders) == 0:
            patrol_points = self.get_patrol_points(game_state)

            # Use the patrol_index to get the current patrol point
            current_patrol_point = patrol_points[self.patrol_index]
            patrol_distance = self.get_maze_distance(my_pos, current_patrol_point)
            features['patrol'] = patrol_distance

            # Check if the agent is at the current patrol point
            if my_pos == current_patrol_point:
                # Move to the next patrol point
                self.patrol_index = (self.patrol_index + 1) % len(patrol_points)
   

        return features
      
    def get_weights(self, game_state, action):
        # Adjust weights to consider the patrol feature
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 
                'average_invader_distance': -5, 'stop': -100, 'reverse': -2, 
                'patrol': -1, 'reverse_last_move': -500}  # Heavy penalty for reversing last move
        
    def get_patrol_points(self, game_state):
        """
        Modify patrol points to start at the bottom of the accessible area and then patrol upwards.
        """
        map_width = game_state.data.layout.width
        map_height = game_state.data.layout.height

        # Determine the frontier line
        if self.red:
            patrol_x = map_width // 2 - 1
        else:
            patrol_x = map_width // 2

        # Create patrol points starting from the bottom
        patrol_points = []
        for y in range(map_height - 2, 0, -1):  # Start from bottom and go upwards
            if not game_state.has_wall(patrol_x, y):
                patrol_points.append((patrol_x, y))

        return patrol_points
      
    def get_dot_density(self, game_state, point, radius=5):
          dot_count = 0
          for x in range(max(1, point[0] - radius), min(game_state.data.layout.width, point[0] + radius + 1)):
              for y in range(max(1, point[1] - radius), min(game_state.data.layout.height, point[1] + radius + 1)):
                  if (x, y) in game_state.get_food():
                      dot_count += 1
          return dot_count




