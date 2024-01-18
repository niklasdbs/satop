#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "TOPEnvState.h"
#include "Resource.h"

namespace py = pybind11;


class ResourceEncoder{
public:
    virtual int encode(TOPEnvState& state, Resource& resource, int offset_0, int offset_1, py::detail::unchecked_mutable_reference<float, 2> target, float walking_time, float distance) = 0;
    int n_features = 0;
};
