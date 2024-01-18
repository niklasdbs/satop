#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <map>
#include "TOPEnvState.h"
#include "graph/Graph.h"
#include "Agent.h"
#include "Config.h"

namespace py = pybind11;

class ObservationEncoder{
public:
    ObservationEncoder(Graph& graph, int number_of_resources, int number_of_actions, Config config);
    virtual ~ObservationEncoder();

    virtual std::map<std::string, py::array_t<float>> encode(TOPEnvState& state, Agent& current_agent) = 0;
    virtual std::map<std::string, std::vector<int>> shape() = 0;
protected:
    Graph& graph;
    py::array_t<float> distance_agent_to_action(TOPEnvState& state, Agent& current_agent);
    float distance_normalization = 3000.0f;//TODO do not hardcode
    const int number_of_resources;
    const int number_of_actions;
    Config config;
};