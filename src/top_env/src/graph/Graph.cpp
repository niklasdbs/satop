#include "graph/Graph.h"
#include <limits>

float Graph::distance(Agent& agent, const Edge& edge){
    //assumes that resource are located at the end of an edge 
    int current_edge_length = agent.position_edge->length;
    int distance_to_current_edge_end = current_edge_length - agent.position_on_edge;

    
    return (shortest_path_lengths[edge.id])[agent.position_node->id] + distance_to_current_edge_end + edge.length;
}

std::tuple<float, float, float, float> Graph::get_min_max_x_y()
{
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y= std::numeric_limits<float>::lowest();

    for (Node* node : nodes){
        if (node->x < min_x){
            min_x = node->x;
        }
        if (node->x > max_x){
            max_x = node->x;
        }

        if (node->y < min_y){
            min_y = node->y;
        }
        if (node->y > max_y){
            max_y = node->y;
        }
    }

    return {min_x, max_x, min_y, max_y};
}