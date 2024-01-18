import numpy as np
from time import time
from omegaconf.dictconfig import DictConfig

from envs.utils import get_distance_matrix

class ACO:
    def __init__(self, config: DictConfig, graph, env) -> None:
        self.number_of_ants = 100000
        self.prob_alpha = 1800.0
        self.alpha = 1.0
        self.beta = 2
        self.max_time = config.max_time #np.inf
        self.computation_time = config.computation_time #np.inf
        self.default_pheromon = 0.01
        self.evaporation_rate = 0.1
        self.distance_matrix = get_distance_matrix(graph) 
        self.speed = config.speed_in_kmh /3.6
        self.resource_id_to_edge_id_mapping = env.get_resource_id_to_edge_id_mapping()
        self.edge_id_to_action_mapping = env.get_edge_id_to_action_mapping()
        
    
    def train(self):
        pass
    
    def act(self, state):
        #state["violations"] [0] = resource id [1] = in_violation, [2] time_last_violations
        violation_resources = [r for r in state["violations"] if r[1] == 1]
        
        violations = [int(r[0]) for r in violation_resources] #contains a list of the resource ids in violation
        
        position = int(state["current_position"][0]) #current position
        violation_times = np.array([r[2] for r in violation_resources]) #time point of the violation for each resource
        
        start_time = int(state["current_time"][0])
                


        #if no violations
        if len(violations) == 0:
            return position


        best_score = -np.inf
        best_solution = None
        
        phero = {}
        self.default_pheromon = 1 / len(violations)


        start_time_computation = time()
        for ant in range(self.number_of_ants):
            path, scores  = self.run_ant(violations, violation_times, phero, start_time, position)
            score = sum(scores)
            
            if score >= best_score:
                best_solution = path
                best_score = score
                

            norm = sum([phero.get(d1, {}).get(d2, self.default_pheromon)
                        for d1 in (position, *violations) for d2 in (position, *violations)])
            for i in range(len(path)):
                d1 = position if i == 0 else path[i-1]
                d2 = path[i]
                old = phero.get(d1, {}).get(d2, self.default_pheromon)
                phero.setdefault(d1, {})[d2] = (1-self.evaporation_rate) * old + scores[i] / norm            
            
            
            if time() - start_time_computation > self.computation_time:
                break
            
        
        
        if best_solution == None:
            raise RuntimeError("no solution found")
        else:
            return self.edge_id_to_action_mapping[self.resource_id_to_edge_id_mapping[best_solution[0]]]
            
            
    def run_ant(self, violations, violation_times, pheromones, start_time, start_pos):
        
        scores = []
        path = []
        
        mask = np.ones(len(violations), dtype=bool)
        
        t = start_time
        pos = start_pos
        
        while sum(mask) > 0:
            travel_times = np.array([self.distance_matrix[pos, r_id] for r_id in violations])/self.speed
            
            time_in_violation = t - violation_times
            assert sum(time_in_violation < 0) == 0
            
            tag_prob = np.exp(- (travel_times + time_in_violation)/self.prob_alpha)
            
            phero = np.array([pheromones.get(pos, {}).get(d, self.default_pheromon) for d in violations])
            prob = np.power(phero, self.alpha) + np.power(tag_prob, self.beta)
            prob *= mask
            prob = prob / prob.sum()

            action = np.random.choice(len(violations), p=prob)
            mask[action] = 0
            path.append(violations[action])
            scores.append(tag_prob[action])            
            
            resource_edge_id = self.resource_id_to_edge_id_mapping[violations[action]]
            t += travel_times[action]
            pos = self.edge_id_to_action_mapping[resource_edge_id]

            if t - start_time > self.max_time:
                break
            
            
        return path, scores