#include "observation_encoders/resource_observations/ResourceObservationEncoder.h"
#include "observation_encoders/resource_observations/ResourceStatusEncoder.h"
#include "observation_encoders/resource_observations/ResourceAgentInfoEncoder.h"
#include "observation_encoders/resource_observations/ResourcePositionEncoder.h"
#include "Config.h"

ResourceObservationEncoder::ResourceObservationEncoder(Graph& graph, float speed_in_ms, Config config): graph(graph), speed_in_ms(speed_in_ms){
    bool do_not_use_fined_status = std::get<int>(config["do_not_use_fined_status"]);
    bool optimistic_in_violation = std::get<int>(config["optimistic_in_violation"]);
    float distance_normalization = std::get<float>(config["distance_normalization"]);
    bool add_x_y_position_of_resource = std::get<int>(config["add_x_y_position_of_resource"]);
    
    resource_encoders.push_back(new ResourceStatusEncoder(do_not_use_fined_status, optimistic_in_violation));
    resource_encoders.push_back(new ResourceAgentInfoEncoder(distance_normalization));

    if (add_x_y_position_of_resource){
        auto [min_x, max_x, min_y, max_y] = graph.get_min_max_x_y();
        
        resource_encoders.push_back(new ResourcePositionEncoder(min_x, max_x, min_y, max_y));
    }

    features_per_resource = 0;

    for (auto* encoder : resource_encoders){
        features_per_resource += encoder->n_features;
    }


}

py::array_t<float> ResourceObservationEncoder::encode_resources(TOPEnvState& state, Agent& current_agent){
    auto resource_observation = py::array_t<float, py::array::c_style>({state.number_of_resources, features_per_resource});
    auto resource_observation_buffer_ptr = resource_observation.mutable_unchecked<2>();


    for (Resource* resource : state.resources){
        int i = resource->id;

        int offset = 0;

        for (ResourceEncoder* encoder : resource_encoders){
            float distance = graph.distance(current_agent, *(resource->edge));
            float walking_time = distance/speed_in_ms;
            offset += encoder->encode(state, *resource, i, offset, resource_observation_buffer_ptr, walking_time, distance);
        }


    }


    return resource_observation;
}
