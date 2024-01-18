#pragma once
#include "graph/Node.h"
#include "graph/Edge.h"
#include <list>

class Agent{
public:
    Agent(int id, const Node* position_node, const Node* position_node_source, int position_on_edge, const Edge* position_edge);
    const int id;
    float reward_since_last_action = 0;
    float discounted_reward_since_last_action = 0;
    int time_last_action = 0;
    const Node* position_node;
    const Node* position_node_source;
    int position_on_edge = 0;
    int current_action = -1;
    void reset_after_current_action_completed();
    bool current_action_complete = false;
    std::list<int> current_route;//TODO
    const Edge* position_edge;
};