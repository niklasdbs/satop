#pragma once
#include <vector>
#include <map>
#include "Edge.h"
#include "Node.h"
#include "Resource.h"
#include "Agent.h"
#include <tuple>
#include <unordered_map>

class Graph{
public:
    /// @brief edge_id (start of edge), node_id (current_position)
    std::unordered_map<int, std::unordered_map<int, std::vector<int>*>> shortest_path_lookup;
    std::unordered_map<int, std::unordered_map<int, int>> shortest_path_lengths;
    std::vector<Edge*> edges;
    std::vector<Node*> nodes;
    float distance(Agent& agent, const Edge& edge);
    /// @brief get min/max x and y
    /// @return min_x, max_x, min_y, max_y
    std::tuple<float, float, float, float> get_min_max_x_y();
};

