#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "TOPEnv.h"
#include <pybind11/stl_bind.h>

namespace py = pybind11;
PYBIND11_MAKE_OPAQUE(std::vector<int>);
//PYBIND11_MAKE_OPAQUE(std::map<std::string, py::array_t<float>>);


//todo look into opaque binding and check if automatic conversion is fast enough https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
PYBIND11_MODULE(top_env, m) {
//        py::bind_map<std::map<std::string, py::array_t<float>>>(m, "ObsMapNp");

py::bind_vector<std::vector<int>>(m, "VectorInt");
//TODO this breaks toch default collate...
//py::bind_map<std::map<std::string, py::array_t<float>>>(m, "ObsMapNp"); 

py::enum_<DataSplit>(m, "DataSplit")
        .value("TRAINING", DataSplit::TRAINING)
        .value("VALIDATION", DataSplit::VALIDATION)
        .value("TEST", DataSplit::TEST)
        .export_values();

py::class_<TOPEnv>(m, "TOPEnv")
 .def(py::init<DataSplit, Config>())
 .def("number_of_actions", &TOPEnv::number_of_actions)
 .def("number_of_agents", &TOPEnv::number_of_agents)
 .def("observation_shape", &TOPEnv::observation_shape)
 .def("current_agent_selection", &TOPEnv::current_agent_selection)
 .def("active_agents", &TOPEnv::active_agents)
 .def("step", &TOPEnv::step)
 .def("reset", &TOPEnv::reset)
 .def("last", &TOPEnv::last)
 .def("get_final_advanced_metrics", &TOPEnv::get_final_advanced_metrics)
 .def("get_resource_id_to_edge_id_mapping", &TOPEnv::get_resource_id_to_edge_id_mapping)
 .def("get_edge_id_to_action_mapping", &TOPEnv::get_edge_id_to_action_mapping);


}

