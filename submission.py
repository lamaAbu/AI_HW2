from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


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


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


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
    
