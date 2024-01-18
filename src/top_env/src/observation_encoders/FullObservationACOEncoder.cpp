#include "observation_encoders/FullObservationACOEncoder.h"
#include <algorithm>

FullObservationACOEncoder::FullObservationACOEncoder(Graph &graph, int number_of_resources, int number_of_actions, Config config) : ObservationEncoder(graph, number_of_resources, number_of_actions, config)
{
    //resource_encoder = new ResourceObservationEncoder(graph, std::get<float>(config["speed_in_kmh"])/3.6f, config); 
}

std::map<std::string, py::array_t<float>> FullObservationACOEncoder::encode(TOPEnvState& state, Agent& current_agent)
{
    int i = 0;
    int action_number = 0;//TODO initial position is not action target
    for (int edge_id : state.action_to_edge_id)
    {
        if(edge_id == current_agent.position_edge->id){
            action_number = i;
            break;
        }
        i++;
    }
    auto agent_position = py::array_t<float, py::array::c_style>(1);
    agent_position.mutable_unchecked()[0] = static_cast<float>(action_number);
    
   // auto current_day = py::array_t<float, py::array::c_style>(1);
    //current_day.mutable_unchecked()[0] = static_cast<float>(state.current_day);
    
    auto current_time = py::array_t<float, py::array::c_style>(1);
    current_time.mutable_unchecked()[0] = static_cast<float>(state.current_time);
    
    auto violations = py::array_t<float, py::array::c_style>({state.number_of_resources, 3});
    //std::fill(resource_history.mutable_data(), resource_history.mutable_data()+resource_history.size(), 0.0f);    
    auto violations_ptr = violations.mutable_unchecked<2>();


    for (auto* resource : state.resources){
        int resource_offset = resource->id;
        int offset = 0;
        
        
        violations_ptr(resource_offset, offset++) = resource->id;
        violations_ptr(resource_offset, offset++) = resource->status == in_violation;
        violations_ptr(resource_offset, offset++) = resource->time_last_violation;

    }

    return {
        {"violations", violations},
  //      {"distance_to_action", distance_agent_to_action(state, current_agent)},
        {"current_position", agent_position},
 //       {"current_day", current_day},
        {"current_time", current_time},
    };
}

std::map<std::string, std::vector<int>> FullObservationACOEncoder::shape(){
    return {
        {"violations", std::vector<int>{number_of_resources, 3}},
//        {"distance_to_action", std::vector<int>{number_of_actions}},
        {"current_position", std::vector<int>{1}},
        {"current_time", std::vector<int>{1}},
    };

}
