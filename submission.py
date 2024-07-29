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



def heuristic_for_minimax(env: WarehouseEnv, agent_id):
    if not env.done():
        return smart_heuristic(env, agent_id)
    
    Lama = env.get_robot(agent_id)
    Jad = env.get_robot(1 - agent_id)

    if Lama.credit < Jad.credit:
        return float('-inf')
    
    elif Lama.credit > Jad.credit:
        return float('inf')
    
    return 0 #?


def time_out_ERROR(start_t, limit_t):
    cur_t = time.time()
    if start_t + limit_t  < cur_t:
        #time limit reached
        raise RuntimeError

def time_out(start_t, limit_t):
    cur_t = time.time()
    if start_t + limit_t  >= cur_t:
        #time is not out yet 
        return False
    return True    

class AgentMinimax(Agent):

    def max_succ(self, env: WarehouseEnv, me_robot, cur_robot, cur_depth, limit_time, start_time):
        max_val = float('-inf')
        operators = env.get_legal_operators(cur_robot)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(cur_robot, op)

        for _, child in zip(operators, children):
            tmp = self.minimax(child, me_robot, cur_robot, cur_depth, limit_time, start_time)
            if tmp > max_val:
                max_val = tmp
        return max_val

    def min_succ(self, env: WarehouseEnv, me_robot, cur_robot, cur_depth, limit_time, start_time):
        min_val = float('inf')
        operators = env.get_legal_operators(cur_robot)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(cur_robot, op)

        for _, child in zip(operators, children):
            tmp = self.minimax(child, me_robot, cur_robot, cur_depth, limit_time, start_time)
            if tmp < min_val:
                min_val = tmp
        return min_val
    
    def minimax(self, env: WarehouseEnv, me_robot, cur_robot, cur_depth, limit_time, start_time):
        time_out_ERROR(start_time, limit_time)
        #from lecture
        #if the state is a terminal state: return the state’s utility
        if env.done() or cur_depth <= 0: #or check time limit and depth?
            return heuristic_for_minimax(env, me_robot)
        
        #if the next agent is MAX: return max-value(state)
        if me_robot == cur_robot:
            return self.max_succ(env, me_robot, cur_robot, cur_depth-1, limit_time, start_time)
        
        #if the next agent is MIN: return min-value(state)
        return self.min_succ(env, me_robot, cur_robot, cur_depth-1, limit_time, start_time)
    

    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        robot_enemy = 1 - agent_id
        depth = 0 #trying
        while True:
            try:
                operators = env.get_legal_operators(agent_id)
                children = [env.clone() for _ in operators]
                for child, op in zip(children, operators):
                    child.apply_operator(agent_id, op)

                children_heuristics = [self.minimax(child, agent_id, robot_enemy, depth, time_limit, start_time) for child in children]
                depth = depth + 3
                max_heuristic = max(children_heuristics)
                index_selected = children_heuristics.index(max_heuristic)
            except RuntimeError:
                return operators[index_selected]

        
        


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
    
