#pragma once
#include <vector>
#include <list>

class AgentSelector{
public:
    AgentSelector(int number_of_agents);
    int next();
    bool is_last();
    void set_agent_needs_to_select_new_action(int agent);
private:
    std::list<int> next_agents;
    int current_agent;
    int selected_agent;
};