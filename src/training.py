import torch

from transformer4bpm import constants, utils, Dataset
from transformer4bpm.args import Args
from transformer4bpm.transformer4bpm_preprocessing.preprocessing_recommendation import PreprocessingRecommendation

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import transformers


args = Args()
if args.preprocess:
    p = PreprocessingRecommendation(args, constants.DATA_PATH_TO_CSV_FILES)
    df = p.preprocess()
else:
    df = dict() # load everything from files
df_train, df_validate, df_test, process_split_dict = utils.train_validate_test_split(df,args)

# load dataset for Trainer
raw_dataset = load_dataset(Dataset.__file__,cache_dir=constants.DATA_CACHE)

# tokenize
tokenized_dataset, tokenizer = utils.tokenize_dataset(args,raw_dataset)

# training
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=20)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.resize_token_embeddings(len(tokenizer))
transformers.logging.set_verbosity_info()
seq2seq_args = Seq2SeqTrainingArguments(
    output_dir = constants.MODELS,
    learning_rate = args.model_params["learning_rate"],
    lr_scheduler_type = "constant",
    evaluation_strategy = "steps",
    eval_steps=1000,
    save_strategy = "steps",
    save_steps=1000,
    num_train_epochs=args.model_params["train_epochs"],
    report_to="tensorboard",
    load_best_model_at_end = True,
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 64
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model,
    seq2seq_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[early_stopping_callback],
)
trainer.train()
