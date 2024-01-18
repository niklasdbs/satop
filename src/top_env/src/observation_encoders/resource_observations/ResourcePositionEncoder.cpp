#include "observation_encoders/resource_observations/ResourcePositionEncoder.h"

ResourcePositionEncoder::ResourcePositionEncoder(float min_x, float max_x, float min_y, float max_y): min_x(min_x), min_y(min_y), max_x(max_x), max_y(max_y)
{
    n_features = 2;
}

int ResourcePositionEncoder::encode(TOPEnvState &state, Resource &resource, int offset_0, int offset_1, py::detail::unchecked_mutable_reference<float, 2> target, float walking_time, float distance)
{
    float x = (resource.x - min_x) / (max_x - min_x);
    float y = (resource.y - min_y) / (max_y - min_y);


    target(offset_0, offset_1++) = x;
    target(offset_0, offset_1++) = y;

    return n_features;
}
