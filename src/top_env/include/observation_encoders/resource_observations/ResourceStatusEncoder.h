#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "TOPEnvState.h"
#include "Resource.h"
#include "ResourceEncoder.h"

namespace py = pybind11;


class ResourceStatusEncoder : public ResourceEncoder{
public:
    ResourceStatusEncoder(bool do_not_use_fined_status, bool optimistic_in_violation);
    int encode(TOPEnvState& state, Resource& resource, int offset_0, int offset_1, py::detail::unchecked_mutable_reference<float, 2> target, float walking_time, float distance) override;
private:
    bool do_not_use_fined_status;
    bool optimistic_in_violation;
};
