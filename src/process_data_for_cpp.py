import hydra
from hydra.utils import to_absolute_path
from omegaconf.dictconfig import DictConfig
import pandas as pd
import numpy as np

from main_tian import load_data_for_env
from tqdm import tqdm


EVENT_TYPE_CONVERSION = {"Arrival": 0,
                         "Departure": 1,
                         "Violation": 2}



def process_graph(graph):
    nodes = set()
    edges = set()
    edge_data = []
    edge_to_edge_id_mapping = {}
    node_to_id_mapping = {}
    
    
    for start_node, end_node, data in graph.edges(data=True):
        nodes.add(start_node)
        nodes.add(end_node)

    i = 0
    for node in nodes:
        node_to_id_mapping[node] = i
        i+= 1

    i = 0
    for start_node, end_node, data in graph.edges(data=True):        
        edge_to_edge_id_mapping[(start_node, end_node)] = i
        edges.add((start_node, end_node))
        start_node_id = node_to_id_mapping[start_node]
        end_node_id = node_to_id_mapping[end_node]
        edge_data.append({
            "edge_id": i,
            "length": data["length"],
            "start_node_id": start_node_id,
            "end_node_id": end_node_id,
        })
        i += 1
    
    node_data = []
    for node, data in graph.nodes(data=True):
        if not node in nodes:
            continue
        
        node_data.append({
            "node_id": node_to_id_mapping[node],
            "x": data["x"],
            "y" : data["y"]
        })
    
    edge_df = pd.DataFrame(edge_data, columns=["edge_id", "length", "start_node_id", "end_node_id"])
    node_df = pd.DataFrame(node_data, columns=["node_id", "x", "y"])

    
    #min_node = min(list(graph.nodes))
    #source_start_node = next(graph.predecessors(min_node))
    #start_edge_id = edge_to_edge_id_mapping[(source_start_node, min_node)]
    
    return nodes, node_to_id_mapping, edges, edge_to_edge_id_mapping, edge_df, node_df

def find_resources(graph, edge_to_edge_id_mapping):
    
    spots_mapping = {}
    resources = []
    
    i = 0
    for start_node, end_node, data in graph.edges(data=True):
        if "spots" in data:
            for spot in data["spots"]:
                
                resource = {
                    "resource_id": i,
                    "x": spot["x"],
                    "y": spot["y"],
                    "edge_id": edge_to_edge_id_mapping[(start_node, end_node)],
                    "position_on_edge": 0 
                }
                
                spots_mapping[spot["id"]] = i
                resources.append(resource)
                
                i += 1
                
    df = pd.DataFrame(resources, columns=["resource_id", "x", "y", "edge_id", "position_on_edge"])
    return df, spots_mapping

def process_events(event_log, spots_mapping):
    spot_ids = set(spots_mapping.keys())
    event_log["DayOfYear"] = event_log["Time"].dt.dayofyear
    relevant_events = event_log[event_log["StreetMarker"].isin(spot_ids)].copy()
    relevant_events["ResourceID"] = relevant_events["StreetMarker"].map(spots_mapping)
    relevant_events["EventType"] = relevant_events["Type"].map(EVENT_TYPE_CONVERSION)
    relevant_events["Year"] = relevant_events["Time"].dt.year
    relevant_events["Month"] = relevant_events["Time"].dt.month
    relevant_events["Hour"] = relevant_events["Time"].dt.hour
    relevant_events["DayOfWeek"] = relevant_events["Time"].dt.dayofweek
    relevant_events["TimeOfDay"] =  ((relevant_events["Time"] - relevant_events["Time"].dt.normalize()) / pd.Timedelta('1 second')).astype(int) #time in seconds since midnight
    relevant_events["TimeStamp"] = relevant_events["Time"].view(np.int64) // 10 ** 9
    # assert relevant_events["TimeStamp"].eq(relevant_events["Time"].map(lambda time: int(time.timestamp()))).all()
    
    df_export = relevant_events[["ResourceID", "TimeStamp", "MaxSeconds", "EventType" ,"Year", "Month", "DayOfYear", "Hour", "TimeOfDay", "DayOfWeek"]]
    
    # resource_events = df_export.groupby("DayOfYear")
    # for day_of_year, group in resource_events:
    #     test =  list(group)
    #     b = len(group)
    #     c = group[group["EventType"] == 2]
    #     d = len(c)
    #     a = 4

    return df_export





def process(graph, event_log, shortest_path_lookup, area_name):
    nodes, node_to_id_mapping, edges, edge_to_edge_id_mapping, edge_df, node_df = process_graph(graph)
    resource_df, spots_mapping = find_resources(graph, edge_to_edge_id_mapping)
    event_df  = process_events(event_log, spots_mapping)
    
    edge_df.to_csv(to_absolute_path(f"../data/2019/{area_name}_edge.csv"))
    node_df.to_csv(to_absolute_path(f"../data/2019/{area_name}_node.csv"))
    resource_df.to_csv(to_absolute_path(f"../data/2019/{area_name}_resources.csv"))
    event_df.to_csv(to_absolute_path(f"../data/2019/{area_name}_event.csv"))
    
    #save memory
    edge_df = None
    node_df = None
    resource_df = None
    event_df = None
    graph = None
    event_log = None

    shortest_path_data = []
    for edge, all_sources in tqdm(shortest_path_lookup.items()):
        for source_node, route in all_sources.items():
            previous_node = route[0]
            route_position = 0
            for current_node in route[1:]:
                #current_edge_id = edge_to_edge_id_mapping.get((previous_node, current_node), edge_to_edge_id_mapping.get((current_node, previous_node), None))
                #assert current_edge_id is not None
                shortest_path_data.append({
                    "target_edge_id": edge_to_edge_id_mapping[edge],
                    "source_node_id": node_to_id_mapping[source_node],
                    "current_edge_id": edge_to_edge_id_mapping[(previous_node, current_node)],
                    #"current_edge_id": current_edge_id,
                    "route_position": route_position
                })
                route_position += 1
                previous_node = current_node
    
    shortest_path_df = pd.DataFrame(shortest_path_data, columns=["target_edge_id", "source_node_id", "current_edge_id","route_position"])
    #shortest_path_data = None
    print("write shortest path to csv")
    shortest_path_df.to_csv(to_absolute_path(f"../data/2019/{area_name}_shortest_path.csv"))

    print("finished")


@hydra.main(config_path="config", config_name="config")
def run(config : DictConfig):
    # import datetime
    # current_day = 2
    # year = 2019
    # start_hour = 7
    # start_time = int(
    #         datetime.datetime.strptime(("%03d %d %02d:00" % (current_day, year, start_hour)),
    #                                     "%j %Y %H:%M").timestamp())
    # current_time = int(
    #         datetime.datetime.strptime(("%03d %d %02d:00" % (current_day, year, 0)),
    #                                     "%j %Y %H:%M").timestamp())

    event_log, graph, shortest_path_lookup = load_data_for_env(config)
    
    process(graph, event_log, shortest_path_lookup, config.area_name)


if __name__ == '__main__':
    run()