#include "observation_encoders/FullObservationGRCNEncoder.h"

FullObservationGRCNEncoder::FullObservationGRCNEncoder(Graph &graph, int number_of_resources, int number_of_actions, Config config) : ObservationEncoder(graph, number_of_resources, number_of_actions, config)
{
    resource_encoder = new ResourceObservationEncoder(graph, std::get<float>(config["speed_in_kmh"])/3.6f, config); 
}

std::map<std::string, py::array_t<float>> FullObservationGRCNEncoder::encode(TOPEnvState& state, Agent& current_agent)
{
    auto resource_observations = resource_encoder->encode_resources(state, current_agent);

    auto current_agent_id = py::array_t<float, py::array::c_style>(1);
    current_agent_id.mutable_unchecked()[0] = static_cast<float>(current_agent.id);

    return {
        {"resource_observations", resource_observations},
        {"distance_to_action", distance_agent_to_action(state, current_agent)},
        {"current_agent_id", current_agent_id},
    };
}

std::map<std::string, std::vector<int>> FullObservationGRCNEncoder::shape(){
    return {
        {"resource_observations", std::vector<int>{number_of_resources, resource_encoder->features_per_resource}},
        {"distance_to_action", std::vector<int>{number_of_actions}},
        {"current_agent_id", std::vector<int>{1}},
    };

}