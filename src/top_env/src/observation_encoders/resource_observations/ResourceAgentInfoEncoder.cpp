#include "observation_encoders/resource_observations/ResourceAgentInfoEncoder.h"
#include <algorithm>

ResourceAgentInfoEncoder::ResourceAgentInfoEncoder(float distance_normalization) : distance_normalization(distance_normalization)
{
    n_features = 5;
}

int ResourceAgentInfoEncoder::encode(TOPEnvState &state, Resource &resource, int offset_0, int offset_1, py::detail::unchecked_mutable_reference<float, 2> target, float walking_time, float distance)
{
    float length_of_day_in_seconds = state.end_of_working_day_time - state.start_time;
    float normalized_current_time = (state.current_time - state.start_time)/length_of_day_in_seconds;

    target(offset_0, offset_1++) = walking_time / 3600.0f;
    target(offset_0, offset_1++) = normalized_current_time;
    target(offset_0, offset_1++) = normalized_current_time + (walking_time / length_of_day_in_seconds);
    target(offset_0, offset_1++) = std::min(2.0f, (resource.status == occupied || resource.status == in_violation) ? (state.current_time + walking_time - resource.arrival_time - resource.max_parking_duration_seconds)/resource.max_parking_duration_seconds : -1.0f);
    target(offset_0, offset_1++) = distance/ distance_normalization;


    return n_features;
}
