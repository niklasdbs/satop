#pragma once
#include "events/Event.h"
#include "TOPEnvState.h"
#include <vector>
#include "Resource.h"
#include "Agent.h"
#include <tuple>
#include "Metrics.h"
#include <optional>
#include "graph/Graph.h"
#include "graph/Node.h"
#include "graph/Edge.h"
#include "AgentSelector.h"
#include <map>
#include <memory>
#include <list>
#include <observation_encoders/ObservationEncoder.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <variant>
#include "Config.h"

namespace py = pybind11;

struct compare_event_pointers {
    bool operator()(const Event *x, const Event *y) const
    {
        if (x->eventTime == y->eventTime){
            return x->eventType>y->eventType;//ensure that agent events come before resource events
        }
        return x->eventTime > y->eventTime ;  
    }
};


enum DataSplit{TRAINING, VALIDATION, TEST};

class TOPEnv {
public:
    explicit TOPEnv(DataSplit split, Config config);
    ~TOPEnv();
    std::variant<bool, std::vector<std::map<std::string, py::array_t<float>>>> reset(bool reset_days, bool only_do_single_episode);
    /// @brief step
    /// @param action id of the edge with resources on it
    /// @return whether the step advanced the environment
    bool step(int action);


    std::tuple<std::map<std::string, py::array_t<float>>, float, float, bool, std::map<std::string, int>> last(int agent_id, bool observe);

    int number_of_actions();
    std::map<std::string, std::vector<int>> observation_shape();
    int number_of_agents();
    std::vector<int> active_agents();
    int current_agent_selection();
    std::map<std::string,float> get_final_advanced_metrics();
    std::vector<int> get_resource_id_to_edge_id_mapping();
    std::map<int, int> get_edge_id_to_action_mapping();
private:
    TOPEnvState* state;

    /// Comparator for events
    compare_event_pointers event_comp;

    /// Contains an event queue (heap) for every episode (e.g. day, year) (do not modify this queues after creation)
    std::vector<std::vector<Event*> *> event_queues;

    /// Event queue (heap) for the current episode group (e.g. day, year) (can be modified) and will be copied by at the start of each episode
    std::vector<Event*> current_event_queue;

    std::map<int, int> day_to_event_queue;

    Graph graph;
    AgentSelector* agent_selector = nullptr;

    const DataSplit data_split;
    ObservationEncoder* observation_encoder;

    int agent_id_selection = -1;

    bool move_other_agents_between_edges = false;
    bool calculate_advanced_metrics;
    bool shared_reward = false;
    bool create_observation_between_steps = false;
    /// @brief remove the days so that a special signal is returned on reset when a complete episode group has been simulated
    bool remove_days = false;
    bool shuffle_days = false;

    bool one_step_actions = false;//TODO do not hardcode
    bool reward_clipping = false;

    float gamma;
    float speed_in_ms;

    int end_hour;
    int start_hour;

    int start_edge_id;

    Metrics metrics;
    std::map<std::string,float> final_advanced_metrics;

    std::vector<float> rewards;
    std::vector<float> discounted_rewards;
    std::vector<bool> dones;
    std::vector<std::shared_ptr<std::map<std::string, int>>> infos;
    std::vector<std::map<std::string, pybind11::array_t<float>>> observations;

    std::list<int> days_to_simulate;

    void init_graph(std::string prefix);
    void init_resources(std::string prefix);
    void init_agents();
    void init_days_to_simulate();
    bool is_in_data_split(int day);
    void init_events(std::string prefix);
    Edge* start_position(int agent_id);

    std::tuple<bool, std::optional<Agent*>> simulate_next_event(bool before_agent_starts);

    bool next_day();
    void reset_resources();
    void handle_events_before_agent_starts();
    void create_events_from_agent_action(Agent* agent, int action);
    void reset_days_in_reset();

};