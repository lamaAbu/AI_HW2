from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random

#added by Lama
from WarehouseEnv import board_size
import time



# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    cur_robot = env.get_robot(robot_id)
    credit = cur_robot.credit

    if env.robot_is_occupied(robot_id):
        # the robot is carrying a package
        destination = cur_robot.package.destination
        return 200 + (10000*(credit + 100) + 1/(100*(1 + manhattan_distance(destination,cur_robot.position))))
    
    #o.w
    else:
        #free robot
        packages_on_board = [package for package in env.packages if package.on_board == True]
        if packages_on_board.__len__() == 0:
            #not possible I guess
            return
        
        #if packages_on_board.__len__() == 1:
        best_package = packages_on_board[0]
        
        if packages_on_board.__len__() == 2:
            if(manhattan_distance(cur_robot.position, best_package.position) > manhattan_distance(cur_robot.position,packages_on_board[1].position)):
                best_package = packages_on_board[1]
 
        return (10000*(credit + 100) + 1/(100*(1 + manhattan_distance(best_package.position,cur_robot.position))))





class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)




def max_succ(children_heuristics):
    max_val = float('-inf')
    for val in children_heuristics:
        if val > max_val:
            max_val = val
    return max_val

def min_succ(children_heuristics):
    min_val = float('inf')
    for val in children_heuristics:
        if val < min_val:
            min_val = val
    return min_val

def time_out(start_t, limit_t):
    cur_t = time.time()
    if start_t + limit_t  >= cur_t:
        #time is not out yet
        return False
    return True

class AgentMinimax(Agent):

    def minimax(self, env: WarehouseEnv, children_heuristics, id, cur_depth):
    #from lecture
        #if the state is a terminal state: return the state’s utility
        if env.done() or cur_depth <= 0: #or check time limit and depth?
            return smart_heuristic(env, id)
        
        #if the next agent is MAX: return max-value(state)
        if id == 0:
            return max_succ(children_heuristics)
        
        #if the next agent is MIN: return min-value(state)
        return min_succ(children_heuristics)
    

    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_heuristics = [self.heuristic(child, agent_id) for child in children]

        best_succ = max(children_heuristics)
        #index_selected = children_heuristics.index(best_succ)
        possible_moves = [i for i, c in enumerate(children_heuristics) if c == best_succ]

        depth = 2*board_size
        index_selected = -1
        #to_apply = None #just initializing
        #while True:
        #    if time_out(start_time, time_limit):
        #        raise RuntimeError
        #    if depth < 0 :
        #        break
        #    to_apply = self.minimax(env, children_heuristics, agent_id,depth)
        #    depth = depth - 3 #but why?
        #    index_selected = children_heuristics.index(to_apply)
        #    
        #if index_selected == -1:
        #    return None
        
        return operators[random.choice(possible_moves)]


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
    
