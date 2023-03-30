import pandas as pd
import networkx as nx
from tqdm import tqdm
from transformer4bpm import syntax
from itertools import product
import matplotlib as plt
import pickle
from transformer4bpm.args import Args

from transformer4bpm import constants, utils



def _model_df_to_model_dict(df: pd.DataFrame)->dict:
    """
    ARGUMENTS:
        df: pd.DataFrame - dataframe containing the parsed elements of all models
    RETURN:
        dict - {model_id: {attributes}, model_id: {attributes},..., model_id: {attributes}} 
    DESCRIPTION:
        Returns a dictionary of all models and their elements.
    """
    models_as_dict = {} # {model_id: {attributes}} 
    count=0
    for k, v in df.groupby('model_id'):
        v=v.set_index('element_id')
        try:
            models_as_dict[k]=v.to_dict(orient='index')
        except ValueError:
            count+=1
            continue
    print("double id"+ str(count))
    return models_as_dict

def _create_graph(process_model_dict: dict, remove_gateways: bool=False, remove_unconnected_nodes: bool=True)->nx.DiGraph:
    """
    ARGUMENTS:
        process_model_dict: dict - dictionary containing the elements of a model
        remove_gateways: bool=False - shall the gateways be removed for the networkx graphs?
        remove_unconnected_nodes: bool=True - shall unconnected nodes be removed for the networkx graphs?
    RETURN:
        nx.Digraph - given model as networkx graph
    DESCRIPTION:
        Creates the networkx graph for a given model. It is possible to drop gateways and unconnected nodes.
    """
    G=nx.DiGraph()
    apply_label_cleaning=['model_name', 'pool', 'lane', 'label']
    items_to_drop_flows = []
    items_to_drop_gateways = []

    # create nodes and edges
    for item_id, item_values in process_model_dict.items():  
        if item_values['category'] in (syntax.NODE_CATEGORIES + [syntax.SEQUENCE_FLOW]):
            for i in apply_label_cleaning:
                item_values[i]=utils.clean(item_values[i])
            G.add_node(item_id, **item_values)         
            for outgoing in item_values['outgoing']:
                if outgoing in process_model_dict: 
                    if process_model_dict[outgoing]["category"] in (syntax.NODE_CATEGORIES + [syntax.SEQUENCE_FLOW]):
                        G.add_edge(item_id,outgoing)
            if item_values['category'] == syntax.SEQUENCE_FLOW:
                items_to_drop_flows.append((item_id,item_values))
            if remove_gateways:
                if item_values['category'] in syntax.GATEWAY_CATEGORIES:
                    items_to_drop_gateways.append((item_id,item_values))

    # drop the sequence flow nodes since they are now edges
    for item_to_drop in items_to_drop_flows:
        item_id = item_to_drop[0]
        item_value = item_to_drop[1]
        predecessors = G.predecessors(item_id)
        successors = G.successors(item_id)
        #print(item_value)
        new_edges = [(p,s, item_value) for p,s in product(predecessors,successors)]
        G.remove_node(item_id)
        G.add_edges_from(new_edges)


    # drop gateways and/or unconnected nodes
    if remove_gateways:
        for item_to_drop in items_to_drop_gateways:  # drops gateways and reconnenct the nodes
            item_id = item_to_drop[0]
            predecessors = G.predecessors(item_id)
            successors = G.successors(item_id)
            new_edges = product(predecessors,successors)
            G.remove_node(item_id)
            G.add_edges_from(new_edges)
    if remove_unconnected_nodes:
        G.remove_nodes_from(list(nx.isolates(G))) # removes unconnected/single nodes
    return G

def draw_graph(graph: nx.DiGraph, with_labels: bool=True, node_description: str='label', edge_description: str='label'):
    """
    ARGUMENTS:
        graph: nx.DiGraph - networkx graph that represents a BPMN model
        with_labels: bool=True - shall the nodes and edges be labeled in the drawing?
        node_description: str='label' - which node attribute serves as node label in the drawing?
        edge_description: str='label' - which edge attribute serves as edge label in the drawing
    DESCRIPTION:
        Creates a drawing of a BPMN model as networkx graph.
    """
    node_labels=nx.get_node_attributes(graph, node_description)
    edge_labels=nx.get_edge_attributes(graph, edge_description)

    options = {
        'node_color': 'black',
        'node_size': 3,
        'width': 1,
        'font_size':6 
    }
    pos_pos={}
    for node in list(graph.nodes.data()):
        pos_pos[node[0]]= node[1]['bounds_x_y']

    nx.draw(graph,pos=pos_pos, with_labels=with_labels, labels=node_labels,**options)
    nx.draw_networkx_edge_labels(
        graph, pos_pos,
        edge_labels=edge_labels,
        font_color='red',
        font_size=6
    )
    plt.axis('off')
    plt.show()

def _list_to_dict(sequence: list, graph: nx.DiGraph, use_exact_seq_len)->dict:
        """
        ARGUMENTS:
            sequence: list - a sequence of nodes as list
            graph: nx.DiGraph - a model as networkx graph
        RETURN:
            dict - a dictionary that contains the (context,target) pair from the given sequence
        DESCRIPTION:
        Generates the (context,target) pairs from a given sequnence of nodes.
        Example: {
            'context': <pool> acme ag <lane> head of human resources <Exclusive_Databased_Gateway> applicant suitable <Task> send rejection <Exclusive_Databased_Gateway>  <EndNoneEvent> 
            'target': applicant rejected
        }
        """
        context_nodes = sequence[:-1]
        target_node = sequence[-1]
        label = nx.get_node_attributes(graph, "label")
        category = nx.get_node_attributes(graph,"category")
        #pool = nx.get_node_attributes(graph,"pool")
        #lane = nx.get_node_attributes(graph,"lane")
        model_id = nx.get_node_attributes(graph,"model_id")
        organization_id = nx.get_node_attributes(graph,"organization_id")
        context = ""
        for c in context_nodes:
            context += "<" + category[c] + "> " + label[c] + " "
        if not use_exact_seq_len:
            context = "complete sequence of length " + str(len(context_nodes)) +": " + context
        #if len(lane[target_node])>0:
        #    context = "<lane> " + lane[target_node] + " " + context
        #if len(pool[target_node])>0:
        #    context = "<pool> " + pool[target_node] + " " + context
        context += "<" + category[target_node] + "> "
        target = graph.nodes[target_node]['label']
        return {'context': " ".join(context.split())+"</s>", 'target': target+"</s>", 'model_id': model_id[target_node], 'organization_id': organization_id[target_node]}

class PreprocessingRecommendation:
    def __init__(self, args: Args, path_to_csv_files: str): 
        """
        ARGUMENTS:
            args: Args - arguments for the recommendation task
            path_to_csv_files: str - path to the csv files that contain the model jsons
        DESCRIPTION:
            Initializes object of class PreprocessingRecommendation.
        """
        self.path_to_csv_files = path_to_csv_files
        self.args = args

    def preprocess(self)->pd.DataFrame:
        """
        RETURN:
            pd.DataFrame - dataframe that contais one sequence per row as (context, target) pairs
        DESCRIPTION:
        Loads the parsed elements of all BPMN models,
        creates networkx graphs of all BPMN models, and
        generates all sequences over all BPMN models and returns them as (context, target) pairs.
        """
        print("Load data.")
        df_parsed_elements = utils.load_data(self.args, self.path_to_csv_files)
        print("Filter data.")
        df_parsed_filtered_elements = utils.filter_data(self.args,df_parsed_elements)
        print("Create networkx graphs from BPMN models.")
        graphs = self._create_networkx_graphs(df_parsed_filtered_elements)
        print("Generate sequences.")
        df_sequences = self.generate_sequences(graphs)
        return df_sequences
        
    def _create_networkx_graphs(self, df_parsed_elements: pd.DataFrame)->list:
        """
        ARGUMENTS:
            df_parsed_elements: pd.DataFrame - dataframe containing the parsed elements of all models
        RETURN:
            list - list containing all models as networkx graphs
        DESCRIPTION:
            Create the networkx graphs of all BPMN models.
            The sequence flows are represented as edges in the networkx graphs.
            The elements of a models are represented as nodes in the networkx graphs and have the following attributes:
            category (= type), label, lane, pool
        """
        pkl_file = constants.DATA_INTERIM / "bpmn_models_as_networkx_graphs.pkl"
        # check if bpmn models have already been transformed into networkx graphs 
        if not self.args.create_nx_graphs:
            # read in the pickle file
            graphs = nx.read_gpickle(pkl_file)
            print("Loaded filtered BPMN as networkx graphs from " + str(pkl_file) + ".")
        else:
            models_as_dict = _model_df_to_model_dict(df_parsed_elements)
            graphs=[]
            for model_id in tqdm(models_as_dict.keys(),desc="Create nx graph for each model: "):
                G = _create_graph(models_as_dict[model_id],remove_gateways=False, remove_unconnected_nodes=True)
                graphs.append(G)#nx.to_dict_of_dicts(G))
            with open(pkl_file, 'wb') as f:
                pickle.dump(graphs, f)
            print("Created filtered BPMN as networkx graphs and saved them in " + str(pkl_file) + ".")
        return graphs

    def generate_sequences(self, graphs: list)->pd.DataFrame:
        """
        ARGUMENTS:
            graphs: list - list containing all models as networkx graphs
        RETURN:
            pd.DataFrame - dataframe that contais one sequence per row as (context, target) pairs
        DESCRIPTION:
        Generates all sequences over all models and returns them as (context, target) pairs
        """
        pkl_file = constants.DATA_INTERIM / "sequences.pkl"
        if not self.args.create_sequences:
            # read in the pickle file
            df = pd.read_pickle(pkl_file)
            print("Loaded sequences from " + str(pkl_file) + ".")
        else:
            sequences = []
            for graph in tqdm(graphs, desc='Create sequences: '):
                for node in graph.nodes():
                    sequences_node = self._get_all_sequences_to_node(node, graph)
                    if len(sequences_node)>0:
                        sequences.extend(sequences_node)
            df = pd.DataFrame(sequences)
            df = df[df["target"]!=""]
            df.to_pickle(pkl_file)
            print("Generated sequences and saved them in " + str(pkl_file) + ".")
        return df

    def _get_all_sequences_to_node(self, node: str, graph: nx.DiGraph)->list:
        """
        ARGUMENTS:
            node: str - a node in model / graph
            graph: nx.DiGraph - a model as networkx graph
        RETURN:
            list - list of sequences that end in the given node
        DESCRIPTION:
        Generates unique sequences over all models and returns them as (context, target) pairs
        """
        sequences = []
        for n in graph:
            all_simple_paths = [p for p in nx.all_simple_paths(graph,n,node,cutoff=self.args.SEQUENCE_LENGTH)]
            category = nx.get_node_attributes(graph,"category")
            all_simple_paths = [p for p in all_simple_paths if category[p[-1]] in syntax.ACTIVITY_CATEGORIES + syntax.EVENT_CATEGORIES]
            if self.args.USE_EXACT_SEQUENCE_LENGTH:
                all_simple_paths = [s for s in all_simple_paths if len(s)==self.args.SEQUENCE_LENGTH+1]
            if len(all_simple_paths)>0:
                sequences.extend(all_simple_paths)
        return [_list_to_dict(s, graph, self.args.USE_EXACT_SEQUENCE_LENGTH) for s in sequences]
