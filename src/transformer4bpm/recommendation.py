from tqdm import tqdm
from transformer4bpm import constants
from more_itertools import chunked
import pandas as pd
import numpy as np
import json
import networkx as nx

def generate_recommendation_cases(args, simulated_models):
    recommendation_cases_pkl_file = constants.RECOMMENDATION_CASES_FILE
    if not args.create_recommendation_cases:
        with open(recommendation_cases_pkl_file,"r") as rcf:
            recommendation_cases = json.load(rcf)["data"]
        print("Loaded recommendation cases from " + str(recommendation_cases_pkl_file) + ".")
    else:
        result_list = []
        for model_id, simulated_models_model_id in tqdm(simulated_models.items(), desc= "Generating recommendation cases:"):
            for simulated_model in simulated_models_model_id:
                graph = nx.from_dict_of_dicts(simulated_model['model'],create_using=nx.DiGraph)
                predecessors = sorted(list(graph.predecessors(simulated_model['target_node'])))
                graph.remove_node(simulated_model['target_node'])
                graph.add_edges_from([(predecessor, 'target_node') for predecessor in predecessors])
                attributes = simulated_model['attributes']
                target_node = simulated_model["target_node"]
                pool = attributes["pool"][target_node]
                lane = attributes["lane"][target_node]
                category = attributes["category"][target_node]
                attributes["pool"]["target_node"] = pool
                attributes["lane"]["target_node"] = lane
                attributes["category"]["target_node"] = category
                result_list.append({
                                'model_id':model_id,
                                'context':nx.to_dict_of_dicts(graph),
                                'predecessors':tuple(predecessors), 
                                'target_node':target_node,
                                'target_label':simulated_model["target_label"],
                                'attributes':attributes,
                                'pool':pool,
                                'lane':lane,
                                'category':category})
        result_df =pd.DataFrame(result_list) 
        result_df=result_df.groupby(['predecessors','model_id','pool','lane','category']).agg({"target_node":list, "target_label":list, "attributes":'first', 'context':'first'}).reset_index()
        recommendation_cases = result_df.to_dict(orient='records')
        with open(recommendation_cases_pkl_file,"w") as rcf:
            json.dump({"data": recommendation_cases},rcf)
        print("Created recommendation cases and saved them in " + str(recommendation_cases_pkl_file) + ".")
    return recommendation_cases


def recommendation(args, recommendation_cases, timestr, model, tokenizer):
    solved_recommendation_cases = [] 
    case_id=0
    for case in tqdm(recommendation_cases,desc="Solve recommendation cases one by one: "):
        # splits each (target, subgraph) dict into N sequences, where N corresponds to all subgraph paths of
        # length == context_len that lead to the target node
        # we want to generate the predicted targets for each context, but then aggregate the result and
        # evaluate each case only once
        input_sequences, labels_in_context = get_recommendation_sequences(args,case)
        predictions = []
        scores = []
        sample_output = []
        inputs = tokenizer(input_sequences,return_tensors='pt',padding=True)
        inputs.to(model.device)
        backup = 9 # backup to prediction list length if generated recommendations are equal after prediction cleaning
        sample_output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_return_sequences=args.prediction_list_length+backup,
            num_beams=args.prediction_list_length+backup,
            no_repeat_ngram_size = 1, # no word repetitions
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
            )
        solved_recommendation_cases.append(recommendation_one_case(args, case, case_id, sample_output, predictions, scores, input_sequences, labels_in_context, tokenizer, backup))
        case_id+=1
    recommendation_file_name = constants.RECOMMENDATION_ROOT / (timestr + ".json")
    with open(recommendation_file_name,"w") as pf:
        json.dump({"solved_recommendation_cases": solved_recommendation_cases}, pf)
    print("Generated recommendations and saved them in " + str(recommendation_file_name) + ".")
    return recommendation_file_name

def recommendation_one_case(args, case, case_id, sample_outputs, predictions, scores, input_sequences, labels_in_context, tokenizer, backup):
    for preds_sequence,scores_sequence,sequence in zip(chunked(sample_outputs["sequences"].cpu(),args.prediction_list_length+backup),chunked(sample_outputs["sequences_scores"].cpu(),args.prediction_list_length+backup),input_sequences):
        preds_sequence = tokenizer.batch_decode(preds_sequence, skip_special_tokens=False)
        preds_sequence = [prediction_cleaning(p) for p in preds_sequence]
        preds_sequence = list(set([p for p in preds_sequence if p!=""]))[:10]
        if len(preds_sequence)<10:
            print("length of rec list"+str(len(preds_sequence)))
        predictions.extend(preds_sequence)
        scores.extend(scores_sequence)
    scores = np.exp(scores)
    recommendations_with_score = [(r,float(s[0])) for r,s in ranking_max(args,predictions, scores)]
    return {"case_id": case_id, 
        	"model_id": case["model_id"], 
            "organization_id": [case["attributes"]["organization_id"][k] for k in [case["attributes"]["organization_id"].keys()][0]][0], 
            "labels_in_context": labels_in_context, 
            "target_labels": case["target_label"],
            "recommendations_with_scores": recommendations_with_score}
    

def get_recommendation_sequences(args,case):
    model = nx.from_dict_of_dicts(case["context"],create_using=nx.DiGraph)
    labels_in_context = []
    for node in model.nodes():
        if node=="target_node":
            continue
        else:
            label = case["attributes"]["label"][node]
            if label=="":
                label = case["attributes"]["category"][node].replace("_","")
            labels_in_context.append(label)
    attributes = case["attributes"]
    sequences = get_all_sequences_to_node(args,"target_node",model, attributes)
    contexts = [s["context"] for s in sequences] # without node to predict
    return contexts, labels_in_context

def get_all_sequences_to_node(args, node, graph, attributes)->list:
    sequences = []
    for n in graph:
        all_simple_paths = [p for p in nx.all_simple_paths(graph,n,node,cutoff=args.SEQUENCE_LENGTH)]
        if args.USE_EXACT_SEQUENCE_LENGTH:
            all_simple_paths = [s for s in all_simple_paths if len(s)==args.SEQUENCE_LENGTH+1]
        if len(all_simple_paths)>0:
            sequences.extend(all_simple_paths)
    return [list_to_dict(s,attributes, args.USE_EXACT_SEQUENCE_LENGTH) for s in sequences]


def ranking_max(args,predictions, scores):
    predictions_dict, maxLen = collect_scores(predictions,scores)
    # save the activities together with their confidences in a list but such that every activity has the same number of confidences (add 0's)
    # also sort the confidences per activity
    predictions_list = []
    if maxLen == 1:
        predictions_list = [(k,v) for k,v in predictions_dict.items()]
    else:
        for k,v in predictions_dict.items():
            v.sort(reverse=True)
            while len(v) < maxLen:
                v.append(0)
            predictions_list.append((k,v))
    # now sort the activites according to their confidences
    ranking = sorted(predictions_list, key=lambda tup: tuple(map(lambda i: tup[1][i],list(range(len(tup[1]))))), reverse=True)
    # if two or more activites have the same confidence, sort them according to their label
    ranking = sorted(ranking, key=lambda tup: (tup[1][0],tup[0]), reverse=True)
    # reduce the sorted list to the top10 activities with their maximum confidences
    return ranking[:args.prediction_list_length]

def collect_scores(predictions,scores):
    predictions_dict = dict()
    maxLen = 1
    for p,s in zip(predictions,scores):
        if p in predictions_dict:
            predictions_dict[p].append(s)
            if len(predictions_dict[p]) > maxLen:
                maxLen = len(predictions_dict[p])
        if p not in predictions_dict:
            predictions_dict[p] = [float(s)]
    return predictions_dict, maxLen


def list_to_dict(sequence: list, attributes,use_exact_seq_len)->dict:
        context_nodes = sequence[:-1]
        target_node = sequence[-1]
        label = attributes["label"]
        category = attributes["category"]
        context = ""
        for c in context_nodes:
            context += "<" + category[c] + "> " + label[c] + " "
        if not use_exact_seq_len:
            context = "complete sequence of length " + str(len(context_nodes)) +": " + context
        context += "<" + category[target_node] + "> "
        return {'context': " ".join(context.split())}


def prediction_cleaning(p):
    p = p.replace("<pad>","").replace("</s>","") 
    p = p.strip()
    p = " ".join(p.split()) 
    return p