#include "observation_encoders/resource_observations/ResourceStatusEncoder.h"

ResourceStatusEncoder::ResourceStatusEncoder(bool do_not_use_fined_status, bool optimistic_in_violation) : do_not_use_fined_status(do_not_use_fined_status), optimistic_in_violation(optimistic_in_violation)
{
    n_features = 4 + (optimistic_in_violation ? 1 : 0);
}

int ResourceStatusEncoder::encode(TOPEnvState& state,Resource& resource, int offset_0, int offset_1, py::detail::unchecked_mutable_reference<float, 2> target, float walking_time, float distance)
{
    int resource_status = resource.status;
    if (do_not_use_fined_status){
        if (resource.status == fined){
            resource_status = ResourceStatus::occupied;
        }
    }

    for (int i = 0; i<4; i++){
        target(offset_0, offset_1 + i) = 0.0f;
    }
    target(offset_0, offset_1 + resource_status) = 1.0f;

    offset_1 += 4;
    

    if (optimistic_in_violation){
        target(offset_0, offset_1++) = resource_status == ResourceStatus::occupied && (state.current_time + walking_time > (resource.arrival_time + resource.max_parking_duration_seconds)) ? 1.0f : 0.0f;
    }

    return n_features;
}