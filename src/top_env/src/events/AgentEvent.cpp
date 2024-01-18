#include "events/AgentEvent.h"

AgentEvent::AgentEvent(int time, int positionNodeID, int positionNodeSourceID, int positionOnEdge, int agentID, bool completesAction, Edge* const position_edge) : 
    Event(time, agent_event) ,positionNodeID(positionNodeID), positionNodeSourceID(positionNodeSourceID), positionOnEdge(positionOnEdge), agentID(agentID), completesAction(completesAction), position_edge(position_edge)
{
}