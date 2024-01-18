#include "TOPEnv.h"
#include "events/AgentEvent.h"
#include "events/ResourceEvent.h"
#include <cassert>
#include <cmath>
#include <iterator>
#include "Utils.h"
#include <set>
#include <queue>
#include "observation_encoders/FullObservationGRCNEncoder.h"
#include "observation_encoders/FullObservationPOTOPEncoder.h"
#include "observation_encoders/FullObservationTimeGraphEncoder.h"
#include "observation_encoders/FullObservationACOEncoder.h"
#include <iostream>
#include <unordered_map>
#ifdef NDEBUG
#define DEBUG(x)
#else
#define DEBUG(x) do { std::cerr << x << std::endl; } while (0)
#endif

TOPEnv::TOPEnv(DataSplit split, Config config) : data_split(split){
    //event_log, graph, shortest_path_lookup
    std::string prefix = "/home/wiss/strauss/projects/satop/data/"; //todo do not hardcode
    prefix += std::to_string(std::get<int>(config["year"]));
    prefix += "/";
    prefix += std::get<std::string>(config["area_name"]);
    prefix += "_"; 
    //set random seed
    std::srand(std::get<int>(config["seed"]));
    start_edge_id = 229; 

    shuffle_days = std::get<int>(config["shuffle_days"]);
    remove_days = (data_split == TRAINING && !shuffle_days) || (data_split == TEST || data_split == VALIDATION);

    reward_clipping = std::get<int>(config["reward_clipping"]);
    if (data_split == TEST || data_split == VALIDATION){
        shuffle_days = false;
        calculate_advanced_metrics = true;
    }
    else{
        calculate_advanced_metrics = false;
    }

    shared_reward = std::get<int>(config["shared_reward"]);
    create_observation_between_steps = std::get<int>(config["create_observation_between_steps"]);
    move_other_agents_between_edges = std::get<int>(config["move_other_agents_between_edges"]);
    gamma = std::get<float>(config["gamma"]);
    end_hour = std::get<int>(config["end_hour"]);
    start_hour = std::get<int>(config["start_hour"]);
    speed_in_ms = std::get<float>(config["speed_in_kmh"])/3.6f;
    init_days_to_simulate();
    int number_of_agents = std::get<int>(config["number_of_agents"]);
    state = new TOPEnvState(number_of_agents);
    init_graph(prefix);
    init_resources(prefix);
    init_events(prefix);
    init_agents();

    std::string observation = std::get<std::string>(config["observation"]);
    if (observation == "FullObservationGRCNSharedAgent")
    {
        observation_encoder = new FullObservationGRCNEncoder(graph, state->number_of_resources, state->action_to_edge_id.size(), config);
    }
    else if (observation == "FullObservationPOTOP")
    {
        observation_encoder = new FullObservationPOTOPEncoder(graph, state->number_of_resources, state->action_to_edge_id.size(), config);   
    }
    else if (observation == "FullObservationTianGraph"){
        observation_encoder = new FullObservationTimeGraphEncoder(graph, state->number_of_resources, state->action_to_edge_id.size(), config);
    }
    else if (observation == "FullObservationACO"){
        observation_encoder = new FullObservationACOEncoder(graph, state->number_of_resources, state->action_to_edge_id.size(), config);
    }
    else {
        throw std::invalid_argument("unkown observation: " + observation);
    }

    DEBUG("Graph: Edges: " << graph.edges.size() << "Nodes: " << graph.nodes.size());
    DEBUG("Number of resources:" << state->number_of_resources);
    DEBUG("Number of actions: " << state->action_to_edge_id.size());
    //how to read stuff/definitions
    //graph:
    //nodes:
    //id(int)
    //edges:
    //id(int),length(float),id_source(int),id_target(int)
    //edge_to_resource
    //edge_id, resource_id
    //shortest paths (only routes to start of edge!)
    //target_edge_id, source_node_id, current_node_id 
    
    //resources
    //resource_id, x, y, edge_id?, position_on_edge?

    //events?
    //resource_id, time_stamp(?), max_seconds, event_type, year, month, day_of_year, hour, time_of_day, day_of_week
}


TOPEnv::~TOPEnv(){
    delete state;
}



void TOPEnv::init_graph(std::string prefix)
{
    //edges, nodes, distances, ...
    //shortest paths need to be stored nodeXnode
    //edge needs to be identified by id and not (u,v) as graph should be undirected
    
    auto edge_df = Utils::read_csv(prefix + "edge.csv");
    int number_of_edges = edge_df.size();
    auto node_df = Utils::read_csv(prefix + "node.csv");
    int number_of_nodes = node_df.size();
    auto shortest_path_df = Utils::read_csv(prefix + "shortest_path.csv");

    std::vector<Edge*> edges = std::vector<Edge*>(number_of_edges);
    std::vector<Node*> nodes = std::vector<Node*>(number_of_nodes);



    for (auto &node_csv : node_df){
        int id = stoi(node_csv[1]);
        float x = stof(node_csv[2]);
        float y = stof(node_csv[3]);

        Node* node = new Node(id, x, y);
        nodes[id] = node;
    }

    for (auto &edge_csv : edge_df){
        int id = stoi(edge_csv[1]);
        float length = stof(edge_csv[2]);
        int start_node_id = stoi(edge_csv[3]);
        int end_node_id = stoi(edge_csv[4]);

        Edge* edge = new Edge(id, length, nodes[start_node_id], nodes[end_node_id]);
        edges[id] = edge;
    }

    graph.nodes = nodes;
    graph.edges = edges;

    std::set<int> target_edge_ids;
    std::set<int> source_node_ids;
    std::unordered_map<int, std::unordered_map<int, int>> size_of_shortest_paths;
    std::unordered_map<int, std::unordered_map<int, int>> length_of_shortest_paths;

    for (auto &shortest_path_csv : shortest_path_df){
        int target_edge_id = stoi(shortest_path_csv[1]);
        target_edge_ids.insert(target_edge_id);
        int source_node_id = stoi(shortest_path_csv[2]);
        source_node_ids.insert(source_node_id);
        ((size_of_shortest_paths[target_edge_id])[source_node_id])++;
        (length_of_shortest_paths[target_edge_id])[source_node_id] += graph.edges[target_edge_id]->length;
    }


    graph.shortest_path_lengths = length_of_shortest_paths;

    for (int target_edge_id : target_edge_ids){
        for (int source_node_id : source_node_ids){
            (graph.shortest_path_lookup[target_edge_id])[source_node_id] = new std::vector<int>((size_of_shortest_paths[target_edge_id])[source_node_id]);
        }
    }


    for (auto &shortest_path_csv : shortest_path_df){
        int target_edge_id = stoi(shortest_path_csv[1]);
        target_edge_ids.insert(target_edge_id);
        int source_node_id = stoi(shortest_path_csv[2]);
        source_node_ids.insert(source_node_id);
        int edge_id = stoi(shortest_path_csv[3]);
        int route_position = stoi(shortest_path_csv[4]);

        (*(graph.shortest_path_lookup[target_edge_id])[source_node_id])[route_position] = edge_id;
    }

// //#ifdef NDEBUG
//     for (auto [target_edge_id, node_map] : graph.shortest_path_lookup){
//         for (auto [source_node_id, route] : node_map){
//             if (route->empty()){
//                 continue;
//             }

//             assert(("route starts wrong", source_node_id == graph.edges[route->front()]->source->id));
//             int previous_edge_id = -1;
//             for (int edge_id : *route){
//                 if (previous_edge_id > 0){
//                     assert(("Invalid routes" ,graph.edges[previous_edge_id]->target->id == graph.edges[edge_id]->source->id));
//                 }

//                 previous_edge_id = edge_id;

//             }
//         }
//     }
// //#endif
}

void TOPEnv::init_resources(std::string prefix)
{
    auto resource_df = Utils::read_csv(prefix+"resources.csv");
    int number_of_resources = resource_df.size();
    state->number_of_resources = number_of_resources;
    state->resources = std::vector<Resource*>(number_of_resources);


    for (auto &resource_csv : resource_df){
        int resource_id = stoi(resource_csv[1]);
        float x = stof(resource_csv[2]);
        float y = stof(resource_csv[3]);
        int edge_id = stoi(resource_csv[4]);
        int position_on_edge = stoi(resource_csv[5]);

        Edge* edge = graph.edges[edge_id];
        Resource* resource = new Resource(resource_id, position_on_edge,edge, x,y);
        state->resources[resource_id] = resource;
        edge->resourceIDs.push_back(resource_id);
    }

    //create action mapping
    for (Edge* edge : graph.edges){
        if (!edge->resourceIDs.empty()){
            state->action_to_edge_id.push_back(edge->id);
        }
    }

}

void TOPEnv::init_events(std::string prefix){
    auto event_df = Utils::read_csv(prefix + "event.csv");
    int i = 0;
    for (int day : days_to_simulate) {
        event_queues.push_back(new std::vector<Event *>());
        day_to_event_queue[day] = i;
        i++;
    }

    auto day_set = new std::set<int>(days_to_simulate.begin(), days_to_simulate.end());
    for (auto &event_csv : event_df)
    {
        ResourceEvent* r_event = new ResourceEvent(event_csv);
        if (!day_set->contains(r_event->day))
        {
            delete r_event;
            continue;
        }

        event_queues[day_to_event_queue[r_event->day]]->push_back(r_event);
    }

    for (int i = 0; i < event_queues.size(); i++) {
        std::make_heap(event_queues[i]->begin(), event_queues[i]->end(), event_comp);
    }

}

void TOPEnv::init_agents()
{   
    for (Agent* agent: state->agents){
        delete agent;
    }

    state->agents = std::vector<Agent*>(state->number_of_agents);
    state->active_agents = std::vector<int>(state->number_of_agents);
    for(int agent_id=0; agent_id< state->number_of_agents; agent_id++){
        Edge* start = start_position(agent_id);

        Agent* agent = new Agent(agent_id, start->target, start->source, int(start->length), start);
        
        state->agents[agent_id] = agent;
        state->active_agents[agent_id] = agent_id;
    }
}

void TOPEnv::init_days_to_simulate(){
    days_to_simulate.clear();
    for (int day =1; day<366; day++){
        if (is_in_data_split(day)){
            days_to_simulate.push_back(day);
        }
    }
}


bool TOPEnv::is_in_data_split(int day){
    if (data_split == TRAINING){
        return day % 13 > 1;
    }
    else if (data_split == VALIDATION){
        return day % 13 == 1;
    }
    else if (data_split == TEST)
    {
        return day % 13 == 0;
    }
    else{
        throw std::exception();
    }
}

Edge* TOPEnv::start_position(int agent_id)
{
    return graph.edges[start_edge_id];
}


std::vector<int> TOPEnv::get_resource_id_to_edge_id_mapping(){
    std::vector<int> r_id_to_e_id_mapping = std::vector<int>(state->number_of_resources);

    for (auto& resource : state->resources){
        r_id_to_e_id_mapping[resource->id] = resource->edge->id;
    }

    return r_id_to_e_id_mapping;

}
    
std::map<int, int>  TOPEnv::get_edge_id_to_action_mapping(){
    std::map<int, int> e_id_to_action_mapping;

    int i = 0;
    for (int edge_id : state->action_to_edge_id){
        e_id_to_action_mapping[edge_id] = i;

        i++;
    }

    return e_id_to_action_mapping;
}


bool TOPEnv::step(int action){
    Agent* agent_that_needs_to_act = state->agents[agent_id_selection];

    //remove agents that are not finished/handle them
    if (dones[agent_id_selection]){
        auto it = std::find(state->active_agents.begin(), state->active_agents.end(), agent_id_selection);
        state->active_agents.erase(it);
        

        return false;
    }
    

    agent_that_needs_to_act->current_action = action;
    agent_that_needs_to_act->time_last_action = state->current_time;

    create_events_from_agent_action(agent_that_needs_to_act, action);

    bool need_to_advance_env = agent_selector->is_last();

    if (need_to_advance_env) {
        std::vector<Agent*> agents_that_completed_their_action;
        int number_of_agents_that_completed_their_action = 0;

        while (true){
            auto [need_to_simulate_more_events, agent_that_completed_action] = simulate_next_event(false);

            if (!agent_that_completed_action.has_value()) {
                //this case means that all agents are finished because the end of the day is reached
                agents_that_completed_their_action = state->agents;
                number_of_agents_that_completed_their_action = agents_that_completed_their_action.size();
                for (Agent* agent : agents_that_completed_their_action){
                    dones[agent->id] = true;
                }
            }
            else if (agent_that_completed_action.value() != nullptr )
            {
                agents_that_completed_their_action.push_back(agent_that_completed_action.value());
                number_of_agents_that_completed_their_action++;
            }

            if (!need_to_simulate_more_events){
                break;
            }
            
        }

        assert(number_of_agents_that_completed_their_action>0);

        if (create_observation_between_steps){
            for (Agent* agent : state->agents){
                if (std::find(agents_that_completed_their_action.begin(), agents_that_completed_their_action.end(), agent) != agents_that_completed_their_action.end()){
                    continue; //the agents that completed their action are handled later
                }
            
                observations[agent->id] = observation_encoder->encode(*state, *agent);
                rewards[agent->id] = reward_clipping ? std::min(1.0f, agent->reward_since_last_action) : agent->reward_since_last_action;
                discounted_rewards[agent->id] = reward_clipping ? std::min(1.0f, agent->discounted_reward_since_last_action) : agent->discounted_reward_since_last_action;
                std::map<std::string, int> infos_tmp = {{"dt",  (state->current_time - agent->time_last_action)}};
                std::shared_ptr<std::map<std::string, int>> shared = std::make_shared<std::map<std::string, int>>(infos_tmp);
                infos[agent->id] = shared;

            }
            

        }


        for (Agent* agent : agents_that_completed_their_action)
        {
            //update agent selector with agents that need to select a new agent in the next step
            agent_selector->set_agent_needs_to_select_new_action(agent->id);

            //create observation, update rewards, ...
            rewards[agent->id] = reward_clipping ? std::min(1.0f, agent->reward_since_last_action) : agent->reward_since_last_action;
            discounted_rewards[agent->id] = reward_clipping ? std::min(1.0f, agent->discounted_reward_since_last_action) : agent->discounted_reward_since_last_action;
            std::map<std::string, int> infos_tmp = {{"dt",  (state->current_time - agent->time_last_action)}};
            std::shared_ptr<std::map<std::string, int>> shared = std::make_shared<std::map<std::string, int>>(infos_tmp);
            infos[agent->id] = shared;

            //reset agent state after current action completed
            agent->reset_after_current_action_completed();

            //create observation for agent when the agent state has been reset
            observations[agent->id] = observation_encoder->encode(*state, *agent);
        }

    }

    agent_id_selection = agent_selector->next();

    return need_to_advance_env;

}

std::tuple<std::map<std::string, py::array_t<float>>, float, float, bool, std::map<std::string, int>> TOPEnv::last(int agent_id, bool observe)
{
    if(agent_id == -1){
        agent_id = agent_id_selection;
        assert(agent_id != -1);
        assert(state->agents[agent_id]->current_action == -1);
    }

    return {observations[agent_id], rewards[agent_id], discounted_rewards[agent_id], dones[agent_id], *infos[agent_id]};
}

int TOPEnv::number_of_actions()
{
    return state->action_to_edge_id.size();
}

std::map<std::string, std::vector<int>> TOPEnv::observation_shape()
{
    return observation_encoder->shape();
}

int TOPEnv::number_of_agents()
{
    return state->number_of_agents;
}

std::vector<int> TOPEnv::active_agents()
{
    return state->active_agents;
}

int TOPEnv::current_agent_selection()
{
    if (state->active_agents.empty()){
        return -1;
    }
    else{
        return agent_id_selection;
    }
}

std::tuple<bool, std::optional<Agent*>> TOPEnv::simulate_next_event(bool before_agent_starts){
    Event* event = current_event_queue.front();

    if (event->eventTime >= state->end_of_working_day_time) {
        return {false, {}};
    }

    std::pop_heap(current_event_queue.begin(), current_event_queue.end(), event_comp);
    current_event_queue.pop_back();

    Event* next_event_after_event_to_simulate = current_event_queue.empty() ? nullptr : current_event_queue.front();

    bool need_to_simulate_more_events = 
                                        event->eventType == resource_event || 
                                                            (!static_cast<AgentEvent*>(event)->completesAction ||
                                                            (next_event_after_event_to_simulate != nullptr && next_event_after_event_to_simulate->eventType == agent_event &&  (next_event_after_event_to_simulate->eventTime - event->eventTime) == 0));

    if (before_agent_starts) {
        assert(("no agent events, before the agents start working", event->eventType == resource_event));

        need_to_simulate_more_events = next_event_after_event_to_simulate != nullptr && next_event_after_event_to_simulate->eventTime < state->start_time;
    }

    int time_diff = event->eventTime - state->current_time;
    assert(("encountered negative time!", time_diff >= 0));


    state->current_time = event->eventTime;
    state->current_hour = event->eventTime/60*60;

    Agent* agent_that_completed_action = nullptr;

    if (event->eventType == agent_event){
        AgentEvent* agent_event = static_cast<AgentEvent*>(event);

        Agent* agent = state->agents[agent_event->agentID];
        int time_diff_since_beginning_of_action = event->eventTime - agent->time_last_action;

        const Node* const previous_position = agent->position_node;
        const Edge* const previous_edge = agent->position_edge;
        //get prev pos

        //update pos
        agent->position_node = graph.nodes[agent_event->positionNodeID];
        agent->position_node_source = graph.nodes[agent_event->positionNodeSourceID];
        agent->position_on_edge = agent_event->positionOnEdge;
        agent->position_edge = agent_event->position_edge;

        assert(agent->position_edge->source->id == previous_position->id);

        if (move_other_agents_between_edges)
        {
            //move them on the edges (they can not move past an intersection)
            for (Agent* other_agent : state->agents){
                if (agent != other_agent){
                    other_agent->position_on_edge += int(time_diff * speed_in_ms);
                }
            }
        }


        bool position_changed = previous_position->id != agent_event->positionNodeID; //because of wait action

        if (position_changed){
            //collect resources 
            //resources are positioned at the end of an edge

            int number_of_fined_resources = 0;

            for (int r_id : agent->position_edge->resourceIDs){
                Resource* resource = state->resources[r_id];

                if (resource->status == in_violation){
                    number_of_fined_resources++;
                    resource->status = fined;


                    if (calculate_advanced_metrics){
                        int time_until_fine = state->current_time - resource->time_last_violation;
                        metrics.time_until_fine.push_back(time_until_fine);
                    }

                   
                }
            }

            metrics.fined_resources += number_of_fined_resources;

            if (shared_reward){
                for (Agent* other_agent : state->agents){
                    int time_diff_since_beginning_of_action_other = event->eventTime - other_agent->time_last_action;

                    other_agent->reward_since_last_action += number_of_fined_resources;
                    other_agent->discounted_reward_since_last_action += pow(gamma, time_diff_since_beginning_of_action_other) * number_of_fined_resources;
                }
             }
             else{
                float reward = static_cast<float>(number_of_fined_resources);
                float discounted_reward = pow(gamma, time_diff_since_beginning_of_action) * reward;

                agent->reward_since_last_action += reward;
                agent->discounted_reward_since_last_action += discounted_reward;
            }


        }

        if (agent_event->completesAction){
            agent_that_completed_action = agent;
        }
        
        delete agent_event;

    }
    else if (event->eventType == resource_event)
    {
        ResourceEvent* const resource_event = static_cast<ResourceEvent*>(event);
        Resource* const resource = state->resources[resource_event->resourceID];

        if (resource_event->resourceEventType == violation && (resource->status == in_violation || resource->status == fined || resource->status == free_))
        {
            //these events are duplicate or due to some events going over mutliple days => we can ignore them
        }
        else{
            //save event history
            resource->event_history.push_back(resource_event);

            if (calculate_advanced_metrics && !before_agent_starts){
                if (resource->status == in_violation || resource->status == fined){
                    assert (resource_event->resourceEventType != violation);

                    int time_in_violation = event->eventTime - resource->time_last_violation;
                    metrics.violation_durations.push_back(time_in_violation);
                    
                    if (resource->status != fined){
                        metrics.violation_durations_non_fined_resources.push_back(time_in_violation);
                    }
                }
            }

            switch (resource_event->resourceEventType)
            {
                case arrival:
                    resource->status = occupied;
                    resource->arrival_time = event->eventTime;
                    resource->max_parking_duration_seconds = resource_event->maxSeconds;
                    break;
                case depature:
                    resource->arrival_time = 0;
                    resource->status = free_;
                    resource->max_parking_duration_seconds = resource_event->maxSeconds;
                    break;
                case violation:
                    resource->status = in_violation;
                    resource->time_last_violation = event->eventTime;
                    if (!before_agent_starts && calculate_advanced_metrics){
                        metrics.cumulative_resources_in_violation++;
                    }
                    break;
            }
            
        }

    }
    if (before_agent_starts && !need_to_simulate_more_events){
        state->current_time = state->start_time;
    }


    return {need_to_simulate_more_events, agent_that_completed_action};


}


void TOPEnv::reset_resources(){
    ResourceEvent* const next_event = static_cast<ResourceEvent*>(current_event_queue.front());
    for (Resource* const resource : state->resources){
        resource->max_parking_duration_seconds = 0;
        resource->time_last_violation = 0;
        resource->status = free_;
        resource->arrival_time = 0;
        
        if (!resource->event_history.empty()){
            delete resource->event_history.front(); //free initial history event
        }
        //clear history
        resource->event_history.clear(); //TODO free resource events that are created only for the history purpose
        // add initial event in history if wanted
        resource->event_history.push_back(new ResourceEvent( 
            resource->id,
            start, //TODO do not hardcode
            0,
            next_event->year,
            next_event->month,
            next_event->day,
            0,
            next_event->dayOfWeek,
            0,
            0
        ));
    }
}

void TOPEnv::handle_events_before_agent_starts()
{
    while (std::get<0>(simulate_next_event(true))){
        
    }
}

void TOPEnv::create_events_from_agent_action(Agent* agent, int action)
{

    //get route and create agent events   
    int target_edge_id = state->action_to_edge_id[action];

    auto route = (graph.shortest_path_lookup[target_edge_id])[agent->position_node->id];
    // add route (and last part) to agent current route
    assert(route->empty() || graph.edges[(*route)[0]]->source->id == agent->position_node->id);
    assert(!route->empty() || graph.edges[target_edge_id]->source->id == agent->position_node->id);
    int time = state->current_time;

    int prev_pos = agent->position_node->id;
    
    if (one_step_actions){
        int edge_id;
        if (route->empty()){
            edge_id = target_edge_id;
        }
        else{
            edge_id = route->front();
        }
        Edge* edge = graph.edges[edge_id];
        assert(prev_pos == edge->source->id);

        int edge_length = edge->length;
        int travel_time = int(edge_length/ speed_in_ms);

        if (travel_time == 0){ //increase of 1 second so that the events do not happen at the same time
            travel_time++;
        }
        time += travel_time;

        AgentEvent* event = new AgentEvent(time, edge->target->id, edge->source->id, travel_time, agent->id, true, edge);

        current_event_queue.push_back(event);
        push_heap(current_event_queue.begin(), current_event_queue.end(), event_comp);

    }
    else{
        for (int edge_id : *route) { 
            Edge* edge = graph.edges[edge_id];

            assert(prev_pos == edge->source->id);

            int edge_length = edge->length;
            int travel_time = int(edge_length/ speed_in_ms);

            if (travel_time == 0){ //increase of 1 second so that the events do not happen at the same time
                travel_time++;
            }
            time += travel_time;

            AgentEvent* event = new AgentEvent(time, edge->target->id, edge->source->id, travel_time, agent->id, false, edge);

            current_event_queue.push_back(event);
            push_heap(current_event_queue.begin(), current_event_queue.end(), event_comp);

            prev_pos = edge->target->id;
        }
        //add target edge end to route
        Edge* target_edge = graph.edges[target_edge_id];
        int edge_length = target_edge->length;
        int travel_time = int(edge_length/ speed_in_ms);
        if (travel_time == 0){ //increase of 1 second so that the events do not happen at the same time
            travel_time++;
        }

        time += travel_time;

        assert(target_edge->source->id == prev_pos);

        //last event needs to indicate that an agent has to act again
        AgentEvent* event = new AgentEvent(time, target_edge->target->id, target_edge->source->id, travel_time, agent->id, true, target_edge);   


        current_event_queue.push_back(event);
        push_heap(current_event_queue.begin(), current_event_queue.end(), event_comp);
    }
}


bool TOPEnv::next_day(){
    if (calculate_advanced_metrics && metrics.cumulative_resources_in_violation >0){
        //reset before simulation
        metrics.soft_reset();
    }

    //all days have been simulated => indicate that a reset should be done
    if (days_to_simulate.empty())
    {
        if (calculate_advanced_metrics){
            final_advanced_metrics = metrics.full_reset();
        }

        return false;
    }


    int day_index = 0;
    if (shuffle_days){
        day_index = std::rand() % days_to_simulate.size();
    }

    auto day_it = days_to_simulate.begin();
    std::advance(day_it, day_index);
    int next_day = *day_it;

    if (remove_days){
        days_to_simulate.erase(day_it);
    }


    state->current_day = next_day;

    state->start_time = start_hour * 3600;
    state->current_time = 0;
    state->current_hour = 0;
    state->end_of_working_day_time = end_hour * 3600;


    current_event_queue = std::vector<Event *>(*event_queues[day_to_event_queue[state->current_day]]);//copy event queue
    assert(!current_event_queue.empty());
    ResourceEvent* const next_event = static_cast<ResourceEvent*>(current_event_queue.front());
    state->current_month = next_event->month;
    state->current_weekday = next_event->dayOfWeek;
    state->current_year = next_event->year;

    return true;
}


void TOPEnv::reset_days_in_reset(){
    //reset days_to_simulate
    init_days_to_simulate();
    //next_day
    next_day();

    if (calculate_advanced_metrics){
        metrics.soft_reset();
    }

}

std::map<std::string, float> TOPEnv::get_final_advanced_metrics()
{
    return final_advanced_metrics;
}

std::variant<bool, std::vector<std::map<std::string, py::array_t<float>>>> TOPEnv::reset(bool reset_days, bool only_do_single_episode){
    if (reset_days){
        reset_days_in_reset();
    }
    else{
        //this means we have simulated a whole year that consists of multiple episodes

        if (!next_day())
        {
            if (only_do_single_episode){
                return false;
            }
            else{
                reset_days_in_reset();
            }
        }
    }

    reset_resources();
    handle_events_before_agent_starts();

    init_agents();

    observations = std::vector<std::map<std::string, py::array_t<float>>>(state->number_of_agents);
    rewards = std::vector<float>(state->number_of_agents, 0.0f);
    discounted_rewards = std::vector<float>(state->number_of_agents, 0.0f);
    dones = std::vector<bool>(state->number_of_agents, false);
    
    infos = std::vector<std::shared_ptr<std::map<std::string, int>>>(state->number_of_agents);

    for (Agent* agent: state->agents){
        observations[agent->id] = observation_encoder->encode(*state, *agent);
        
        std::map<std::string, int> infos_tmp = {{"dt",  0}};
        std::shared_ptr<std::map<std::string, int>> shared = std::make_shared<std::map<std::string, int>>(infos_tmp);
        infos[agent->id] = shared;

    }


    if (agent_selector != nullptr){
        delete agent_selector;
    }

    agent_selector = new AgentSelector(state->number_of_agents);
    agent_id_selection = agent_selector->next();

    return observations;
}