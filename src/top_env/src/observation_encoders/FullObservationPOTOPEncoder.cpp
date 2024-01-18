#include "observation_encoders/FullObservationPOTOPEncoder.h"
#include "Resource.h"

FullObservationPOTOPEncoder::FullObservationPOTOPEncoder(Graph &graph, int number_of_resources, int number_of_actions, Config config) : ObservationEncoder(graph, number_of_resources, number_of_actions, config)
{
    resource_encoder = new ResourceObservationEncoder(graph, std::get<float>(config["speed_in_kmh"])/3.6f, config); 
}

std::map<std::string, py::array_t<float>> FullObservationPOTOPEncoder::encode(TOPEnvState& state, Agent& current_agent)
{
    auto resource_observations = resource_encoder->encode_resources(state, current_agent);

    auto current_agent_id = py::array_t<float, py::array::c_style>(1);
    current_agent_id.mutable_unchecked()[0] = static_cast<float>(current_agent.id);

    auto [resource_history, resource_history_lengths] = encode_resource_history(state, current_agent);

    return {
        {"resource_observations", resource_observations},
        {"distance_to_action", distance_agent_to_action(state, current_agent)},
        {"current_agent_id", current_agent_id},
        {"resource_history", resource_history},
        {"resource_history_lengths", resource_history_lengths}
    };
}

std::map<std::string, std::vector<int>> FullObservationPOTOPEncoder::shape(){
    return {
        {"resource_observations", std::vector<int>{number_of_resources, resource_encoder->features_per_resource}},
        {"distance_to_action", std::vector<int>{number_of_actions}},
        {"current_agent_id", std::vector<int>{1}},
        {"resource_history", std::vector<int>{number_of_resources, 42, 15}}, //TODO
        {"resource_history_lengths", std::vector<int>{number_of_resources, 1}}
    };

}

std::tuple<py::array_t<float>, py::array_t<float>> FullObservationPOTOPEncoder::encode_resource_history(TOPEnvState& state, Agent& current_agent){
    bool cyclical_time_encoding = true;//TODO
    int max_sequence_length = 42;//TODO
    int features_per_resource = 5 + (cyclical_time_encoding ? 10 : 0);//TODO
    auto resource_observation = py::array_t<float, py::array::c_style>({state.number_of_resources,max_sequence_length, features_per_resource});
    auto resource_observation_buffer_ptr = resource_observation.mutable_unchecked<3>();
    auto sequence_lengths = py::array_t<float, py::array::c_style>({state.number_of_resources,1});
    auto sequence_lengths_buffer_ptr = sequence_lengths.mutable_unchecked<2>();

    for(Resource* const resource : state.resources)
    {
        int resource_offset = resource->id;
        int seq_length = resource->event_history.size();
        auto resource_event_it = resource->event_history.begin();
        if (seq_length > max_sequence_length){
            std::advance(resource_event_it, seq_length-max_sequence_length);
            sequence_lengths_buffer_ptr(resource_offset, 0) = max_sequence_length;
        }
        else{
            sequence_lengths_buffer_ptr(resource_offset, 0) = seq_length;
        }

        int sequence_offset = 0;
        for (auto end = resource->event_history.end(); resource_event_it != end; resource_event_it++) //TODO check
        {
            ResourceEvent* const resource_event = *resource_event_it;
            int event_offset = 0;
            
            for (int i = 0; i<3; i++){
                resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset + i) = 0.0f;
            }

            resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset + resource_event->resourceEventType) = 1.0f;
            event_offset += 3;


            resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = resource_event->maxSeconds/3600.0f;
            resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = resource_event->time_of_day/3600.0f;

            if (cyclical_time_encoding){
                const float pi = 3.14159265358979323846f;
                resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::sin(2*pi*resource_event->hour/23.0f);
                resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::cos(2*pi*resource_event->hour/23.0f);

                //starts with 1
                resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::sin(2*pi*resource_event->day/365.0f);
                resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::cos(2*pi*resource_event->day/365.0f);

                //starts with 1
                resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::sin(2*pi*resource_event->month/12.0f);
                resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::cos(2*pi*resource_event->month/12.0f);

                //starts with 0
                resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::sin(2*pi*resource_event->dayOfWeek/6.0f);
                resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::cos(2*pi*resource_event->dayOfWeek/6.0f);
                
                //time
                resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::sin(2*pi*resource_event->time_of_day/24*60*60.0f);
                resource_observation_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::cos(2*pi*resource_event->time_of_day/24*60*60.0f);

            }


            sequence_offset++;
        }
    }

    return {resource_observation, sequence_lengths};
}