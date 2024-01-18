import numpy as np
from envs.utils import get_distance_matrix

class GreedySingle:
    def __init__(self, config, graph, env) -> None:
        self.speed = config.speed_in_kmh /3.6
        self.distance_matrix = get_distance_matrix(graph) 
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


        
        overstayed = start_time - violation_times
        distance = np.array([self.distance_matrix[position, r_id] for r_id in violations])

        assert not (overstayed < 0).any()
        r_id = violations[np.argmax(-(overstayed + distance / self.speed))]
        return self.edge_id_to_action_mapping[self.resource_id_to_edge_id_mapping[r_id]]

    
