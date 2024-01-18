#include "AgentSelector.h"
#include <cassert>
#include <algorithm>

AgentSelector::AgentSelector(int number_of_agents)
{
    for (int i = 0; i<number_of_agents; i++){
        next_agents.push_back(i);
    }
}
int AgentSelector::next()
{
    int agent_id = next_agents.front();
    next_agents.pop_front();
    return agent_id;
}

bool AgentSelector::is_last()
{
    return next_agents.empty();
}

void AgentSelector::set_agent_needs_to_select_new_action(int agent)
{
    //assert(!(std::find(next_agents.begin(), next_agents.end(), agent) != my_list.end()));//agent is not already in the list
    assert(agent>=0);
    next_agents.push_back(agent);

}
