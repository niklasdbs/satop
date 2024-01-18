#pragma once
#include "graph/Edge.h"
#include <vector>
#include "events/ResourceEvent.h"

enum ResourceStatus{free_, occupied, in_violation, fined};

class Resource{
public:
    Resource(int id, int position_on_edge, Edge* edge, float x, float y);
    const int id;
    ResourceStatus status = free_;
    int time_last_violation = 0;
    int arrival_time = 0;
    int max_parking_duration_seconds = 0;
    const float x;
    const float y;
    const int position_on_edge;
    const Edge* edge;
    std::vector<ResourceEvent*> event_history;
};