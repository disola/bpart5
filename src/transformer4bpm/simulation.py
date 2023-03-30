from transformer4bpm import constants, syntax
import copy
import networkx as nx
import json
from tqdm import tqdm 

### Simulating Process Models ###

def simulate_full_breadth(args, split):
    simulated_models_pkl_file = constants.SIMULATED_MODELS_FILE
    if not args.create_simulations:
        with open(simulated_models_pkl_file, "r") as smf:
            simulated_models = json.load(smf)
        print("Loaded simulated models from " + str(simulated_models_pkl_file) + ".")
    else:
        simulated_models = dict()
        pkl_file = constants.DATA_INTERIM / "bpmn_models_as_networkx_graphs.pkl"
        graphs = nx.read_gpickle(pkl_file)
        for graph in tqdm(graphs,desc="Simulate models under development from model: "):
            if len(graph)==0:
                continue
            process_id = nx.get_node_attributes(graph,"model_id")
            nodes = list(graph.nodes())
            model_id = process_id[nodes[0]]
            if (model_id not in split):
                continue
            node_type = nx.get_node_attributes(graph,"category")
            start_nodes = get_start_nodes(graph)
            # we only want to predict activity and event labels
            nodes_to_predict = [(node,data["label"]) for (node,data) in graph.nodes(data=True) if node_type[node] in (syntax.ACTIVITY_CATEGORIES + syntax.EVENT_CATEGORIES)]
            for (target_node,target_label) in nodes_to_predict:
                if target_label == "":
                    continue
                remaining_nodes = set()
                len_shortest_path, shortest_path = get_shortest_path_to(target_node,start_nodes,graph)
                if len_shortest_path==1000:
                    continue
                if args.USE_EXACT_SEQUENCE_LENGTH:
                    if (len_shortest_path-1)<args.SEQUENCE_LENGTH:
                        continue
                else:
                    if (len_shortest_path-1)<args.min_seq_eval:
                        continue
                remaining_nodes.update(shortest_path)
                for node in graph.nodes():
                    all_simple_paths_to_node = nx.all_simple_paths(graph,shortest_path[0],node,cutoff=len_shortest_path-1) # alle Pfade, die auf der gleichen "Höhe" wie der target Knoten sind
                    for path in all_simple_paths_to_node:
                        remaining_nodes.update(path)
                remaining_graph = copy.deepcopy(graph.subgraph(remaining_nodes))
                attributes={"model_id": process_id, 
                            "category": node_type, 
                            "label": nx.get_node_attributes(graph,"label"),
                            "pool" : nx.get_node_attributes(graph,"pool"),
                            "lane" : nx.get_node_attributes(graph,"lane"),
                            "organization_id" : nx.get_node_attributes(graph,"organization_id")}
                simulated_model = {"target_node": target_node, "target_label": target_label, "model": nx.to_dict_of_dicts(remaining_graph), 'attributes': attributes}
                if model_id in simulated_models:
                    simulated_models[model_id].append(simulated_model)
                else:
                    simulated_models[model_id] = [simulated_model]
        with open(simulated_models_pkl_file, "w") as smf:
            json.dump(simulated_models,smf)
        print("Created simulated models and saved them in " + str(simulated_models_pkl_file) + ".")
    return simulated_models

def get_start_nodes(graph):
    return [node for node in graph.nodes() if graph.in_degree(node)==0 and graph.out_degree(node)>0]


def get_shortest_path_to(target_node,start_nodes,graph):
    len_shortest_path = 1000
    shortest_path = []
    for start_node in start_nodes:
        try:
            shortest_path_from_start_node = nx.shortest_path(graph,source = start_node, target=target_node)
        except nx.NetworkXNoPath:
            shortest_path_from_start_node = []
        len_shortest_path_from_start_node = len(shortest_path_from_start_node)
        if len_shortest_path_from_start_node>0 and len_shortest_path_from_start_node<len_shortest_path:
            len_shortest_path = len_shortest_path_from_start_node
            shortest_path = shortest_path_from_start_node
    return len_shortest_path, shortest_path

def write_simulated_models(simulated_models,file_name):
    with open(file_name,"w") as tc_file:
        for simulated_model in simulated_models:
            write_simulated_model(tc_file,simulated_model)

def write_simulated_model(file,simulated_model):
    attributes = simulated_model["attributes"]
    target_node = simulated_model["target_node"]
    graph = simulated_model["model"]
    label = attributes["label"]
    model_id = attributes["model_id"]
    file.write("Model: " + model_id[list(graph.nodes())[0]] +"\n")
    file.write("Target node: " + str(target_node) + " (" + label[target_node] + ")\n")
    file.write("Nodes: ")
    for (node, data) in graph.nodes(data=True):
        if node == target_node:
            label = "target_node"
        else:
            label = data["label"]
        file.write(str(node) + " (" + label + "), ")
    file.write("\n")
    file.write("Number of nodes: " + str(graph.number_of_nodes()) + ", Number of edges: " + str(graph.number_of_edges()) + "\n")
    file.write("\n")

def get_simulated_model_string(simulated_model):
    attributes = simulated_model["attributes"]
    target_node = simulated_model["target_node"]
    graph = nx.from_dict_of_dicts(simulated_model["model"])
    label = attributes["label"]
    target_string = "label(" + str(target_node) + "," + label[target_node] + ")\t"
    after_string = ""
    for edge in graph.edges():
        after_string += "after("+str(edge[0])+","+str(edge[1])+")\t"
    label_string = ""
    insame_string = ""
    for node in graph.nodes():
        if node != target_node:
            node_label = label[node]
            label_string += "label(" + str(node) + "," + node_label +")\t"
        for node2 in graph.nodes():
            if node2!=node:
                insame_string += "inSameProcess(" + str(node) + "," + str(node2) + ")\t"
    string = target_string + after_string + insame_string + label_string
    string = string.replace(str(target_node), "target_node")
    string = string.replace(str(label[target_node]), "target_label")
    return "\t".join(sorted(list(string.split("\t"))))

