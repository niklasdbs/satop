#include "observation_encoders/ObservationEncoder.h"

ObservationEncoder::ObservationEncoder(Graph &graph, int number_of_resources, int number_of_actions, Config config) : graph(graph), number_of_resources(number_of_resources), number_of_actions(number_of_actions), config(config)
{
}

ObservationEncoder::~ObservationEncoder()
{
    
}

py::array_t<float> ObservationEncoder::distance_agent_to_action(TOPEnvState& state, Agent & current_agent)
{
    auto distance_to_action = py::array_t<float, py::array::c_style>(state.action_to_edge_id.size());

    auto distance_to_action_buffer = distance_to_action.mutable_unchecked<1>();

    int i=0;
    for (int edge_id : state.action_to_edge_id){
        distance_to_action_buffer(i++) = graph.distance(current_agent, *graph.edges[edge_id])/distance_normalization;
    }

    return distance_to_action;
}