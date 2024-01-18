#pragma once
#include "Resource.h"
#include "Agent.h"
#include <vector>

class TOPEnvState{
public:
    TOPEnvState(int number_of_agents);
    ~TOPEnvState();

    int number_of_resources;
    int number_of_agents;
    int end_of_working_day_time;
    int current_time;
    /// Current day
    int current_day = 0;

    /// Current day of week
    int current_weekday = 0;

    /// Current month
    int current_month = 0;

    /// Current year
    int current_year = 0;

    int current_hour = 0;

    int start_time;

    std::vector<Resource*> resources;
    std::vector<Agent*> agents;
    std::vector<int> active_agents;
    std::vector<int> action_to_edge_id;

};