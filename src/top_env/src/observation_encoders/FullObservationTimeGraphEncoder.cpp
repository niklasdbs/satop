#include "observation_encoders/FullObservationTimeGraphEncoder.h"
#include <algorithm>

FullObservationTimeGraphEncoder::FullObservationTimeGraphEncoder(Graph &graph, int number_of_resources, int number_of_actions, Config config) : ObservationEncoder(graph, number_of_resources, number_of_actions, config)
{
    resource_encoder = new ResourceObservationEncoder(graph, std::get<float>(config["speed_in_kmh"])/3.6f, config); 
}

std::map<std::string, py::array_t<float>> FullObservationTimeGraphEncoder::encode(TOPEnvState& state, Agent& current_agent)
{
    auto resource_observations = resource_encoder->encode_resources(state, current_agent);
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
    
    auto current_day = py::array_t<float, py::array::c_style>(1);
    current_day.mutable_unchecked()[0] = static_cast<float>(state.current_day);
    
    auto current_time = py::array_t<float, py::array::c_style>(1);
    current_time.mutable_unchecked()[0] = static_cast<float>(state.current_time);


    //auto [resource_history, resource_history_lengths] = encode_history(state, current_agent);
    auto current_resource_event = encode_current_resource_event(state, current_agent);
    return {
        {"resource_observations", resource_observations},
        {"distance_to_action", distance_agent_to_action(state, current_agent)},
        {"current_position", agent_position},
       // {"resource_history", resource_history},
        //{"resource_history_lengths", resource_history_lengths},
        {"current_day", current_day},
        {"current_time", current_time},
        {"current_resource_event", current_resource_event},
    };
}

std::map<std::string, std::vector<int>> FullObservationTimeGraphEncoder::shape(){
    return {
        {"resource_observations", std::vector<int>{number_of_resources, resource_encoder->features_per_resource}},
        {"distance_to_action", std::vector<int>{number_of_actions}},
        {"current_position", std::vector<int>{1}},
        //{"resource_history", std::vector<int>{number_of_resources, 50, 17}}, 
        //{"resource_history_lengths", std::vector<int>{number_of_resources}},
        {"current_day", std::vector<int>{1}},
        {"current_time", std::vector<int>{1}},
        {"current_resource_event", std::vector<int>{number_of_resources, 18}},
    };

}

py::array_t<float> FullObservationTimeGraphEncoder::encode_agent(TOPEnvState& state, Agent& current_agent)
{
    auto agent_observation = py::array_t<float, py::array::c_style>({state.number_of_resources,1});
    auto agent_observation_buffer_ptr = agent_observation.mutable_unchecked<2>();

    return agent_observation;
}

py::array_t<float> FullObservationTimeGraphEncoder::encode_current_resource_event(TOPEnvState& state, Agent& current_agent)
{
    bool cyclical_time_encoding = true;/
    int features_per_resource = 8 + (cyclical_time_encoding ? 10 : 0);


    auto resource_history = py::array_t<float, py::array::c_style>({state.number_of_resources, features_per_resource});
    std::fill(resource_history.mutable_data(), resource_history.mutable_data()+resource_history.size(), 0.0f);    
    auto resource_history_buffer_ptr = resource_history.mutable_unchecked<2>();

    for (auto* resource : state.resources){
        int resource_offset = resource->id;
        ResourceEvent* prev_event = resource->event_history.back();

        int event_offset = 0;

        //set type of the event 

        ResourceEventType event_type;

        if (resource->status == free_){
            event_type = depature;
        }
        else if (resource->status == occupied)
        {
            event_type = arrival;
        }
        else if (resource->status == in_violation || resource->status == fined){
            event_type = violation;
        }
        

        resource_history_buffer_ptr(resource_offset, event_offset + event_type) = 1.0f;
        event_offset += 4;

        //indicate that this is the last event
        resource_history_buffer_ptr(resource_offset, event_offset + event_type) = 1.0f;

        resource_history_buffer_ptr(resource_offset, event_offset++) = resource->max_parking_duration_seconds/3600.0f;
        resource_history_buffer_ptr(resource_offset, event_offset++) = state.current_time/3600.0f;//TODO maybe event time?

        //time since last event
        resource_history_buffer_ptr(resource_offset, event_offset++) = (state.current_time - prev_event->eventTime) / 3600.0f;


        if (cyclical_time_encoding){
            const float pi = 3.14159265358979323846f;
            resource_history_buffer_ptr(resource_offset, event_offset++) = std::sin(2*pi*state.current_hour/24.0f);
            resource_history_buffer_ptr(resource_offset, event_offset++) = std::cos(2*pi*state.current_hour/24.0f);

            //starts with 1
            resource_history_buffer_ptr(resource_offset, event_offset++) = std::sin(2*pi*(state.current_day-1)/365.0f);
            resource_history_buffer_ptr(resource_offset, event_offset++) = std::cos(2*pi*(state.current_day-1)/365.0f);

            //starts with 1
            resource_history_buffer_ptr(resource_offset, event_offset++) = std::sin(2*pi*(state.current_month-1)/12.0f);
            resource_history_buffer_ptr(resource_offset, event_offset++) = std::cos(2*pi*(state.current_month-1)/12.0f);

            //starts with 0
            resource_history_buffer_ptr(resource_offset, event_offset++) = std::sin(2*pi*state.current_weekday/7.0f);
            resource_history_buffer_ptr(resource_offset, event_offset++) = std::cos(2*pi*state.current_weekday/7.0f);
            
            //time
            resource_history_buffer_ptr(resource_offset, event_offset++) = std::sin(2*pi*state.current_time/(24*60*60.0f));
            resource_history_buffer_ptr(resource_offset, event_offset++) = std::cos(2*pi*state.current_time/(24*60*60.0f));

        }


    }


    return resource_history;
}


std::tuple<py::array_t<float>, py::array_t<float>> FullObservationTimeGraphEncoder::encode_history(TOPEnvState & state, Agent & current_agent)
{
    bool cyclical_time_encoding = true;/
    int max_history_length = 50;
    int features_per_resource = 7 + (cyclical_time_encoding ? 10 : 0);

    auto resource_history = py::array_t<float, py::array::c_style>({state.number_of_resources,max_history_length, features_per_resource});
    std::fill(resource_history.mutable_data(), resource_history.mutable_data()+resource_history.size(), 0.0f);    
    auto resource_history_buffer_ptr = resource_history.mutable_unchecked<3>();

    auto sequence_lengths = py::array_t<float, py::array::c_style>({state.number_of_resources});
    auto sequence_lengths_buffer_ptr = sequence_lengths.mutable_unchecked<1>();



    for (auto* resource : state.resources){
        int resource_offset = resource->id;

        int seq_length = resource->event_history.size();
    
        auto hist_it = resource->event_history.begin();
    
        if (resource->event_history.size() > max_history_length){
            std::advance(hist_it, resource->event_history.size()-max_history_length);
            sequence_lengths_buffer_ptr(resource_offset) = max_history_length;
        }
        else{
            sequence_lengths_buffer_ptr(resource_offset) = seq_length;
        }

        int sequence_offset = 0;

        ResourceEvent* prev_event = nullptr;

        for (auto end = resource->event_history.end(); hist_it != end; ++hist_it)

            ResourceEvent* const resource_event = *hist_it;
            int event_offset = 0;

            //set type of the event
            resource_history_buffer_ptr(resource_offset,sequence_offset, event_offset + resource_event->resourceEventType) = 1.0f;
            event_offset += 4;

            resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = resource_event->maxSeconds/3600.0f;
            resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = resource_event->time_of_day/3600.0f;

            if (prev_event != nullptr){
                //time since last event
                resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = (resource_event->eventTime - prev_event->eventTime) / 3600.0f;
            }


            if (cyclical_time_encoding){
                const float pi = 3.14159265358979323846f;
                resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::sin(2*pi*resource_event->hour/24.0f);
                resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::cos(2*pi*resource_event->hour/24.0f);

                //starts with 1
                resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::sin(2*pi*(resource_event->day-1)/365.0f);
                resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::cos(2*pi*(resource_event->day-1)/365.0f);

                //starts with 1
                resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::sin(2*pi*(resource_event->month-1)/12.0f);
                resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::cos(2*pi*(resource_event->month-1)/12.0f);

                //starts with 0
                resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::sin(2*pi*resource_event->dayOfWeek/7.0f);
                resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::cos(2*pi*resource_event->dayOfWeek/7.0f);
                
                //time
                resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::sin(2*pi*resource_event->time_of_day/24*60*60.0f);
                resource_history_buffer_ptr(resource_offset, sequence_offset, event_offset++) = std::cos(2*pi*resource_event->time_of_day/24*60*60.0f);

            }


            sequence_offset++;
            prev_event = resource_event;

        }

    }


    return {resource_history, sequence_lengths};

}
