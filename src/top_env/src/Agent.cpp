#include "Agent.h"

Agent::Agent(int id, const Node *position_node, const Node *position_node_source, int position_on_edge, const Edge* position_edge)
 : id(id), position_node(position_node), position_node_source(position_node_source), position_on_edge(position_on_edge), position_edge(position_edge)
{
    reset_after_current_action_completed();
}

void Agent::reset_after_current_action_completed()
{
    time_last_action = 0;
    current_action_complete = false;
    current_action = -1;
    current_route.clear();
    reward_since_last_action = 0.0f;
    discounted_reward_since_last_action = 0.0f;
}