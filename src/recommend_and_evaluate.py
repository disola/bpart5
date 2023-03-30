import time
import torch

from transformer4bpm import constants, utils, simulation, recommendation, Dataset, evaluation
from transformer4bpm.args import Args
from transformer4bpm.transformer4bpm_preprocessing.preprocessing_recommendation import PreprocessingRecommendation

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM

args = Args()

df = dict() 
df_train, df_validate, df_test, process_split_dict = utils.train_validate_test_split(df,args)
split_name = "test"
split = process_split_dict[split_name]
simulated_models = simulation.simulate_full_breadth(args, split)

m = constants.MODELS / "t5small_finetuned/checkpoint-138000"
model = AutoModelForSeq2SeqLM.from_pretrained(m)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

timestr = time.strftime("%Y%m%d-%H%M%S")

raw_dataset = load_dataset(Dataset.__file__,cache_dir=constants.DATA_CACHE)
tokenized_dataset, tokenizer = utils.tokenize_dataset(args,raw_dataset)

recommendation_cases = recommendation.generate_recommendation_cases(args,simulated_models)
recommendation_file_name = recommendation.recommendation(args, recommendation_cases, timestr, model, tokenizer)
evaluation.compute_metrics(recommendation_file_name)
