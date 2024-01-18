#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "TOPEnvState.h"
#include "Resource.h"
#include "ResourceEncoder.h"
#include "ResourceStatusEncoder.h"
#include <vector>
#include "graph/Graph.h"
#include "Agent.h"
#include "Config.h"

namespace py = pybind11;


class ResourceObservationEncoder{
public:
    ResourceObservationEncoder(Graph& graph, float speed_in_ms, Config config);
    py::array_t<float> encode_resources(TOPEnvState& state, Agent& current_agent);
    int features_per_resource;
private:
    std::vector<ResourceEncoder*> resource_encoders;
    Graph& graph;
    float speed_in_ms;
};