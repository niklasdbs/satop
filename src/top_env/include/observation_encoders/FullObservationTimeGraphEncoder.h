#pragma once
#include "ObservationEncoder.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <map>
#include "TOPEnvState.h"
#include "resource_observations/ResourceObservationEncoder.h"
#include "Agent.h"
#include "Config.h"
#include <tuple>

namespace py = pybind11;


class FullObservationTimeGraphEncoder : public ObservationEncoder{
public:
    FullObservationTimeGraphEncoder(Graph& graph, int number_of_resources, int number_of_actions, Config config);
    std::map<std::string, py::array_t<float>> encode(TOPEnvState& state, Agent& current_agent) override;
    std::map<std::string, std::vector<int>> shape() override;
private:
    ResourceObservationEncoder* resource_encoder;
    py::array_t<float> encode_agent(TOPEnvState& state, Agent& current_agent);
    py::array_t<float> encode_current_resource_event(TOPEnvState & state, Agent & current_agent);
    std::tuple<py::array_t<float>, py::array_t<float>> encode_history(TOPEnvState& state, Agent& current_agent);
};