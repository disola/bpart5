import random
import numpy as np
import json
import pandas as pd
from transformers import AutoTokenizer
from transformer4bpm.detector import ModelLanguageDetector
from transformer4bpm import constants, syntax,parser
from tqdm import tqdm
import re
import networkx as nx

def load_data(args, path_to_csv_files)->pd.DataFrame:
    """
    RETURN:
        pd.Dataframe - dataframe containing the parsed elements
    DESCRIPTION:
        Parses the model elements from the csv files
        or loads the parsed model elements from a pickle file.
    """
    def _get_lane_label(dict_element_id_to_category_label_parent: dict, parent_id: str)->str:
        """
        ARGUMENTS:
            dict_element_id_to_category_label_parent: dict - a dictionary that contains for each element id the category, the label and the parent of the element as a tuple
            parent_id: str - the parent id of the element for which we want to identify the label of the lane it belongs to
        RETURN:
            str - label of the lane the element belongs to
        DESCRIPTION:
            Checks if an element belongs to a lane and returns the label if so.
        """
        try:
            parent_tuple=dict_element_id_to_category_label_parent[parent_id]
            # if the parent is a lane, then we return the label of the parent/lane
            if parent_tuple[0]=='Lane':
                return parent_tuple[1]
            # otherwise, we return None
            else:
                return None
        except KeyError:
            return None

    def _get_parent_pool_label(dict_element_id_to_category_label_parent: dict, parent_id: str):
        """
        ARGUMENTS:
            dict_element_id_to_category_label_parent: dict - a dictionary that contains for each element id the category, the label and the parent of the element as a tuple
            parent_id: str - the parent id of the element for which we want to identify the label of the lane it belongs to
        RETURN:
            str - label of the pool the element belongs to
        DESCRIPTION:
            Checks if an element belongs to a lane and returns the label if so.
        """
        try:
            parent_tuple=dict_element_id_to_category_label_parent[parent_id]
            # if the parent is already a pool, we return the label of the parent/pool
            if parent_tuple[0]=='Pool':
                return parent_tuple[1]
            # if the parent is a lane, then we have to go one level deeper
            elif parent_tuple[0]=='Lane':
                grand_parent_tuple=dict_element_id_to_category_label_parent[parent_tuple[2]]
                if grand_parent_tuple[0]=='Pool':
                    return grand_parent_tuple[1]
            else:
                return None
        except KeyError:
            return None

    def _get_upper_left_from_bounds(bounds: dict)->list:
        """
        ARGUMENTS:
            bounds: dict - the bounds of an element in a model as given by the json 
        RETURN:
            list - x and y values of the upper left bound 
        DESCRIPTION:
            Returns the x and y values of an element's upper left bound in a model (for visualization as networkx graph).
        """
        x = bounds['lowerRight']['x']
        y = bounds['lowerRight']['y']
        return [x,y]

    pkl_file = constants.DATA_INTERIM / "bpmn_elements_for_networkx.pkl"
    # check if the elements of the bpmn models have already been parsed and saved in a pickle file
    if not args.parse_bpmn_elements_for_networkx:
        # read in the pickle file
        df_parsed_elements = pd.read_pickle(pkl_file)
        print("Loaded parsed BPMN models from " + str(pkl_file) + ".")
    else:
        # parse the elements into a dataframe and pickle it for later usage
        csv_paths = parser.get_csv_paths(path_to_csv_files)
        model_parser = parser.BpmnModelParser(parse_outgoing=True, parse_parent=True, parse_bounds=True, parse_model_name=True)
        df_parsed_elements = model_parser.parse_model_elements(csv_paths)
        df_parsed_elements.reset_index(inplace=True)
        # create a dict that saves for each element_id a tuple with:
        # 0. category/type of the element
        # 1. label of the element
        # 2. parent id of the element
        dict_element_id_to_category_label_parent = dict(zip(df_parsed_elements['element_id'], zip(df_parsed_elements['category'], df_parsed_elements['label'], df_parsed_elements['parent'])))
        # generate some useful additional information for the elements
        df_parsed_elements['lane']=df_parsed_elements['parent'].apply(lambda x: _get_lane_label(dict_element_id_to_category_label_parent,x))
        df_parsed_elements['pool']=df_parsed_elements['parent'].apply(lambda x: _get_parent_pool_label(dict_element_id_to_category_label_parent,x))
        df_parsed_elements['bounds_x_y']=df_parsed_elements['bounds'].apply(lambda x: _get_upper_left_from_bounds(x))
        df_parsed_elements.to_pickle(pkl_file)
        print("Parsed BPMN models and saved them in " + str(pkl_file) + ".")
    return df_parsed_elements


def clean(label: str)->str:
    """
    ARGUMENTS:
        label: str - a string that represents the label of a model element
    RETURN:
        str - cleaned label
    DESCRIPTION:
        Cleans the given label of a model's element (make lower case, formatting, ...).
    """
    if type(label)==str: 
        # handle some special cases
        label = label.replace("-\n","")
        label = label.replace("\n", " ").replace("\r", "")
        label = label.replace("(s)", "s")
        # turn any non alphanumeric characters into whitespace
        label = re.sub("[^A-Za-z0-9]"," ",label)
        # delete unnecessary whitespaces
        label = label.strip()
        label = re.sub("\s{1,}"," ",label)
        # make all lower case
        label = label.lower()
        return label
    else:
        return ''

def train_validate_test_split(df, args):
    if args.do_splitting=="new_split":
        ids = df[args.splitting].unique()
        random.shuffle(ids)
        train, validate, test = np.split(ids, [int(len(ids)*0.85), int(len(ids)*0.925)])
        df_train = df[df[args.splitting].isin(train)].reset_index(drop=True).sample(frac=1)
        df_validate = df[df[args.splitting].isin(validate)].reset_index(drop=True).sample(frac=1)
        df_test = df[df[args.splitting].isin(test)].reset_index(drop=True).sample(frac=1)
        process_split_dict = {"train": list(train), "validate": list(validate), "test": list(test)}
        if args.splitting == "organization_id":
            for split_name in ["train", "validate", "test"]:
                processes = set([model_id for model_id, org_id in zip(df.model_id,df.organization_id) if org_id in process_split_dict[split_name]])
                process_split_dict[split_name] = list(processes)
        df_to_json(df_train,constants.TRAIN_SEQUENCES)
        df_to_json(df_validate,constants.VALIDATE_SEQUENCES)
        df_to_json(df_test,constants.TEST_SEQUENCES)
        with open(constants.PROCESSES_SPLIT_FILE, "w") as f:
            json.dump(process_split_dict,f)
    elif args.do_splitting=="load_sequences_from_jsons":
        df_train = df_from_json(constants.TRAIN_SEQUENCES)
        df_validate = df_from_json(constants.VALIDATE_SEQUENCES)
        df_test = df_from_json(constants.TEST_SEQUENCES)
        with open(constants.PROCESSES_SPLIT_FILE, "r") as f:
            process_split_dict = json.load(f)
    elif args.do_splitting=="generate_sequences_from_split":
        with open(constants.PROCESSES_SPLIT_FILE, "r") as f:
            process_split_dict = json.load(f)
        df_train = df[df["model_id"].isin(process_split_dict["train"])].reset_index(drop=True).sample(frac=1)
        df_validate = df[df["model_id"].isin(process_split_dict["validate"])].reset_index(drop=True).sample(frac=1)
        df_test = df[df["model_id"].isin(process_split_dict["test"])].reset_index(drop=True).sample(frac=1)
        df_to_json(df_train,constants.TRAIN_SEQUENCES)
        df_to_json(df_validate,constants.VALIDATE_SEQUENCES)
        df_to_json(df_test,constants.TEST_SEQUENCES)
    for file_name, split_df in zip(["train","valid","test"],[df_train,df_validate,df_test]):
        print("Number of sequences in " + file_name + " split: " + str(len(split_df)))
    return df_train, df_validate, df_test, process_split_dict

def df_to_json(df,file_name):
    with open(file_name, "w") as f:
        json.dump({'data':df.to_dict(orient='records')},f)

def df_from_json(file_name):
    with open(file_name, "r") as f:
        df = json.load(f)
        df = df["data"]
    return df
    
def preprocess_function(tokenizer, example):
    input =  example["context"]
    target =  example["target"]
    model_inputs = tokenizer(input, 
                    padding = "max_length",
                    max_length = 512,
                    return_tensors = "pt",
                    truncation=True) 
    target_encoding  = tokenizer(target,
                    padding = "max_length",
                    max_length = 128, 
                    return_tensors = "pt",
                    truncation=True) 
    labels = target_encoding.input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels#["input_ids"]
    return model_inputs


def tokenize_dataset(args,raw_dataset):# tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_params["model"]) #,model_max_length=args.model_params["model_max_length"]) # Padding does not make sense here since we apply the tokenizer not batchwise here. We set model max length for truncation.
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': args.additional_special_tokens})
    print("Number of added special tokens: " + str(num_added_tokens))
    return raw_dataset.map(lambda example: preprocess_function(tokenizer, example), batched=True), tokenizer

# FILTERING
def model_filter(df,
    model_id:list=None,
    organization_id:list=None,
    activity:tuple=None,
    event:tuple=None,
    gateway:tuple=None,
    misc:tuple=None,
    model_name:list=None,
    label_length:tuple=None,
    is_example_model:bool=None,
    language:list=None,
    activity_event:tuple=None,
    node:tuple=None
    ):
    if model_id:
        df = df[df["model_id"].isin(model_id)]
    if organization_id:
        df = df[df["organization_id"].isin(organization_id)]
    if activity:
        df = df[df['activity'].between(activity[0],activity[1])]
    if event:
        df = df[df['event'].between(event[0],event[1])]
    if gateway:
        df = df[df['gateway'].between(gateway[0],gateway[1])]
    if misc:
        df = df[df['misc'].between(misc[0],misc[1])]
    if model_name:
        df = df[df["model_name"].isin(model_name)]
    if label_length:
        df = df[df['label_length'].between(label_length[0],label_length[1])]
    if not is_example_model:
        df = df[~df["is_example_model"]]
    if language:
        df = df[df["language"].isin(language)]
    if activity_event:
        df = df[df['activity_event'].between(activity_event[0],activity_event[1])]
    if node:
        df = df[df['node'].between(node[0],node[1])]
    return list(df["model_id"].unique())

def filter_data(args,df_parsed_elements):
    pkl_file = constants.BPMN_MODEL_META_DATA
    print(pkl_file)
    if pkl_file.is_file() and not args.create_model_meta_data:
        df_meta_data = pd.read_pickle(pkl_file)
        print("Loaded meta data of BPMN models (language, number of nodes,...) from " + str(pkl_file) + ".")
    else:
        print('Generate meta data:')
        df_meta_data = create_model_meta_data(df_parsed_elements,pkl_file)
        
    filtered_model_ids = model_filter(df_meta_data,**args.model_filter_options)
    df_parsed_filtered_elements = df_parsed_elements[df_parsed_elements["model_id"].isin(filtered_model_ids)]
    return df_parsed_filtered_elements

def mapping(category, category_mapping):
    if category in category_mapping.keys():
        return category_mapping[category]
    else:
        return 'misc'

def get_length_label(label):
        try:
            if len(clean(label))==0:
                return np.nan
            else:
                return len(clean(label))
        except:
            return np.nan

def create_model_meta_data(df,pkl_file):
    # model_id, model_name, language, nr_activity, nr_event, nr_gateway, nr_misc

    # 01 | map category column to activity, event or gateway
    category_mapping=dict()
    category_mapping.update(dict(zip(syntax.ACTIVITY_CATEGORIES,['activity']*len(syntax.ACTIVITY_CATEGORIES))))
    category_mapping.update(dict(zip(syntax.EVENT_CATEGORIES,['event']*len(syntax.EVENT_CATEGORIES))))
    category_mapping.update(dict(zip(syntax.GATEWAY_CATEGORIES,['gateway']*len(syntax.GATEWAY_CATEGORIES))))

    df['category'] = df['category'].apply(lambda x: mapping(x,category_mapping))

    # 02 | add column with label length to dataframe
    df['label_length']=df['label'].apply(lambda x: get_length_label(x))

    # 03 | create new dataframe 
    # category count per model
    df['count']=1
    df_ = pd.pivot_table(df, values='count', index='model_id', columns='category', aggfunc=np.sum).reset_index()
    # label aggregation per model
    df['label']=df['label'].apply(lambda x: "" if x==None else x )
    df_label = df[['model_id', 'label']].groupby('model_id').agg({'label':" ".join}).reset_index()
    # min label length per model
    df_label_length = df[['model_id', 'label_length']].groupby('model_id').min().reset_index()
    # add model name
    df_model_name = df[['model_id', 'model_name']].groupby('model_id').first().reset_index()
    # add organisation id
    df_organization_id = df[['model_id', 'organization_id']].drop_duplicates()

    # merge all dataframes into df_
    df_=df_.merge(df_label,on='model_id').merge(df_model_name,on='model_id').merge(df_label_length,on='model_id').merge(df_organization_id, on='model_id')

    # add example processes column
    with open(constants.EXAMPLE_PROCESSES) as f:    
        examples = json.load(f)
    example_names = []
    for batch in examples["example_processes"]:
        example_names.extend(batch["content"])
    example_names = set(example_names)

    df_['is_example_model']=df_['model_name'].apply(lambda x: True if x in example_names else False)
    df_['is_example_model'].value_counts()

    # add aggregated counts
    df_['activity_event']=df_['activity']+df_['event']
    df_['node']=df_['activity_event']+df_['gateway']

    # add language column
    tqdm.pandas()
    d = ModelLanguageDetector()
    df_['language'] = df_['label'].progress_apply(lambda x: d._get_text_language(x))
   
    df_.to_pickle(pkl_file)
    print("Generated meta data of BPMN models (language, number of nodes,...) and saved them in " + str(pkl_file) + ".")
    return df_
