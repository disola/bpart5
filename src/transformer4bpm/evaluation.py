import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm
import evaluate
import pandas as pd

def compute_metrics(recommendation_file_name):
    with open(recommendation_file_name,"r") as rf:
        solved_recommendation_cases = json.load(rf)["solved_recommendation_cases"]
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    cosine_loss = tf.keras.losses.CosineSimilarity()
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    records = []
    final_result = {
        "hits_at": np.zeros(10),
        "meteor": np.zeros(10),
        "bleu": np.zeros(10),
        "cosine": np.zeros(10)
    }
    for case in tqdm(solved_recommendation_cases,desc="Evaluate recommendation cases one by one"):
        target_labels = case["target_labels"]
        recommendations = [r[0] for r in case["recommendations_with_scores"]]
        rec  = compute_metrics_records(set(target_labels), recommendations, bleu, meteor, embed, cosine_loss, final_result)
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    recommendation_file_name = str(recommendation_file_name).replace(".json", "_evaluated.json")
    with open(recommendation_file_name,"w") as ef:
        write_metrics(ef, df, final_result)


def compute_metrics_records(targets: set, ranked_preds: list, bleu, meteor, embed, cosine_loss, final_result):
    pred_hits_at = np.zeros(len(ranked_preds))
    for j in range(len(pred_hits_at)):
        pred_hits_at[j] = int(any(x in targets for x in ranked_preds[:(j + 1)]))
    result = dict()
    try:
        result["hits_at"] = pred_hits_at.tolist()
        final_result["hits_at"]+=result["hits_at"]
        # meteor, bleu and cosine
        result["meteor"] = np.zeros(len(ranked_preds))
        result["bleu"] = np.zeros(len(ranked_preds))
        result["cosine"] = np.zeros(len(ranked_preds))
        metric_per_pos = {"meteor": [], "bleu": [], "cosine": []}
        for i in range(len(result["meteor"])):
            if result["hits_at"][i]==1:
                metric_per_pos["bleu"].append(1)
                metric_per_pos["meteor"].append(1)
            else:
                bleu_result = bleu.compute(predictions=[ranked_preds[i]], references=[list(targets)]) ## unigram
                metric_per_pos["bleu"].append(bleu_result["bleu"])
                meteor_result = meteor.compute(predictions=[ranked_preds[i]], references=[list(targets)]) ## ground truth len=1 word
                metric_per_pos["meteor"].append(meteor_result["meteor"])
            result["bleu"][i] = max(metric_per_pos["bleu"])
            result["meteor"][i] = max(metric_per_pos["meteor"])
            metric_per_pos["cosine"].append(get_cosine_loss(ranked_preds[i],list(targets),embed,cosine_loss))
            result["cosine"][i] = max(metric_per_pos["cosine"])
        for m in ["meteor","bleu","cosine"]:
            final_result[m] += result[m]
            result[m] = result[m].tolist()
    except ZeroDivisionError:
        print("zero divison error given by")
        print("targets="+str(targets))
        print("ranked_preds="+str(ranked_preds))
        result["hits_at"] = np.zeros(len(ranked_preds)).tolist()
        result["meteor"] = np.zeros(len(ranked_preds)).tolist()
        result["bleu"] = np.zeros(len(ranked_preds)).tolist()
        result["cosine"] = np.zeros(len(ranked_preds)).tolist()
    return result

def write_metrics(pf, df, final_result):
    result_per_record = df.to_dict(orient="index")
    for k,v in final_result.items():
        final_result[k]=v/len(df)
        if k in ["hits_at","meteor","bleu","cosine"]:
            final_result[k] = final_result[k].tolist()
    final_result["Number of recommendation cases"] = str(len(df))
    d = {"results per case": result_per_record, "final result": final_result}
    json.dump(d,pf)


def get_cosine_loss(prediction,targets,embed,cosine_loss):
    cosines = []
    for i in range(len(targets)):
        cosines.append(-cosine_loss(embed([prediction]),embed([targets[i]])).numpy())
    return max(cosines)

