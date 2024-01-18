#pragma once
#include "Event.h"
#include "graph/Edge.h"

class AgentEvent: public Event {
public:
    AgentEvent(int time, int positionNodeID, int positionNodeSourceID, int positionOnEdge, int agentID, bool completesAction, Edge* const position_edge);
    const int positionNodeID;
    const int positionNodeSourceID;
    const int positionOnEdge;
    const int agentID;
    const bool completesAction;
    Edge* const position_edge;
};