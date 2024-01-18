#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "TOPEnvState.h"
#include "Resource.h"
#include "ResourceEncoder.h"

namespace py = pybind11;


class ResourcePositionEncoder : public ResourceEncoder{
public:
    ResourcePositionEncoder(float min_x, float max_x, float min_y, float max_y);
    int encode(TOPEnvState& state, Resource& resource, int offset_0, int offset_1, py::detail::unchecked_mutable_reference<float, 2> target, float walking_time, float distance) override;
private:
    float min_x;
    float min_y;
    float max_x;
    float max_y;
};
