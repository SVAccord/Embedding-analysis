# -*- coding: utf-8 -*-

"""
Training the Bert (HF) model from scratch. Data must be prepared.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

#import wandb # Removed the use of wandb due to restrictions in Russia.

import torch
print(torch.__version__)
from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModel,
    AutoTokenizer,
)
from tokenizers import BertWordPieceTokenizer, Regex, normalizers

from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm
import pynvml
import json

def save_list_to_txt(my_list, filename):
  """
  Function that stores training parameters on disc 
  """
  try:
    with open(filename, 'w', encoding='utf-8') as file:
      for item in my_list:
        file.write(str(item) + '\n')  
  except Exception as e:
    print(f"exception {e}")

pynvml.nvmlInit()
handle_0 = pynvml.nvmlDeviceGetHandleByIndex(0)
handle_1 = pynvml.nvmlDeviceGetHandleByIndex(1)
info_0 = pynvml.nvmlDeviceGetMemoryInfo(handle_0)
info_1 = pynvml.nvmlDeviceGetMemoryInfo(handle_1)
print(f"GPU0 memory occupied: {info_0.used//1024**2} MB.")
print(f"GPU1 memory occupied: {info_1.used//1024**2} MB.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

LIMIT_DATASET = None 
RANDOM_SEED = 23 
NUM_TOKENIZER_TRAINING_ITEMS = 1_000_000  
VOCAB_SIZE = 32_768  
MODEL_MAX_SEQ_LEN = 512  
DEVICE_BATCH_SIZE = 32 
gradient_accumulation_steps = 4 
batch_size = DEVICE_BATCH_SIZE * gradient_accumulation_steps
NUM_EPOCHS = 8 

header = time.strftime('%Y%m%d-%H%M')
#run = wandb.init(project="pt_BERT_paper_ru_news_99-2009", name=header, entity='my-co')

DATA_DIR = Path("/home/sergeev/Datasets")
MAIN_DIR = Path("/home/sergeev/PythonScripts/NLP/Bert/ru_newspappers_2015-2019")
RUN_DIR = Path("/home/sergeev/PythonScripts/NLP/Bert/ru_newspappers_2015-2019") / f"run_{header}"
TRAINER_HISTORY_PATH = RUN_DIR / "trainer_history.json"
CHECKPOINT_DIR = RUN_DIR / "training_checkpoints"
MODEL_DIR = RUN_DIR / "model"
TOKENIZER_PATH = MAIN_DIR / "tokenizer.json"

RUN_DIR.mkdir(exist_ok=True, parents=True)
train = load_dataset("csv", data_files= str(DATA_DIR) + '//ru_newspappers_2015-2019_0to1M_2_512seqlen.csv', split='train[:95%]') # num_rows: 623846
validation = load_dataset("csv", data_files= str(DATA_DIR) + '//ru_newspappers_2015-2019_0to1M_2_512seqlen.csv', split='train[95%:]')

print(train)
print(validation)
train  = train['text']
validation = validation['text']
print(f" train len {len(train)}")
print(f' validation len {len(validation)}')

# Prepare tokenizator
tokenizer = BertWordPieceTokenizer()
tokenizer._tokenizer.normalizer = normalizers.Sequence(
   [
       normalizers.Replace(Regex("(``|'')"), '"'),
       normalizers.NFD(),
       #normalizers.NFKD(),
       normalizers.Lowercase(),
       normalizers.StripAccents(),
       normalizers.Replace(Regex(" {2,}"), " "), 
       #normalizers.Replace(Regex(r"[^\x00-\x7F]+"), ""), 
       normalizers.Replace(Regex(r"[^\u0000-\u007F\u0400-\u04FF]+"), ""),                             
   ]
) 

def list_to_text(data, separator):
   return separator.join(data)

def text_to_file(text, filename="output.txt"):
   with open(filename, "w", encoding="utf-8") as f:  
       f.write(text)

text_to_file(list_to_text(train, " "), str(MAIN_DIR) + "/train_plain.txt")   
text_to_file(list_to_text(train, " "), str(MAIN_DIR) + "/validation_plain.txt")

special_tokens = [
 "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
files_to_learn = [str(MAIN_DIR)+"/train_plain.txt", str(MAIN_DIR)+"/validation_plain.txt"]
start = time.time()
tokenizer.train(files=files_to_learn, vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
print(f"Tokenizer trained in {(time.time() - start)/60} min.")
tokenizer.save(str(TOKENIZER_PATH)) # Tokenizer trained in 31.452609153588615 min.


model_config = BertConfig(
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=MODEL_MAX_SEQ_LEN,
    attention_probs_dropout_prob=0,  
    hidden_dropout_prob=0,
    output_hidden_states=True,
    device_map="auto",
)

model = BertForMaskedLM(model_config)
print(model.num_parameters())

tokenizer = BertTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
tokenizer.save_vocabulary(save_directory = str(MAIN_DIR))

class TokenizedDataset(torch.utils.data.Dataset):
    "This wraps the dataset and tokenizes it, ready for the model"

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.tokenizer.encode(
            self.dataset[i],
            return_tensors="pt",
            truncation=True,
            max_length=MODEL_MAX_SEQ_LEN - 2,
            padding="max_length",
            return_special_tokens_mask=True,
        )[0, ...]


train_tokenized = list(TokenizedDataset(train, tokenizer))
val_tokenized = list(TokenizedDataset(validation, tokenizer))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    return_tensors="pt",
)

training_args = TrainingArguments(
    learning_rate=2e-4,
    warmup_ratio=0.1,
lr_scheduler_type="cosine", 
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=gradient_accumulation_steps,
optim="adamw_torch",
weight_decay=0.001,
fp16=not torch.cuda.is_bf16_supported(),
bf16=torch.cuda.is_bf16_supported(),
    dataloader_num_workers=4,
    save_steps=1000,
    save_total_limit=100, 
    logging_steps=100, 
    do_train = True,
    do_eval = True,
eval_strategy='steps', # unsl25
eval_steps=500,
    output_dir=CHECKPOINT_DIR,
    #report_to="wandb", # forbidden 403
    report_to=None,
gradient_checkpointing=True,
seed=RANDOM_SEED,
resume_from_checkpoint=True,    
    torch_compile=True, # optimizations
    #optim="adamw_torch_fused", # improved optimizer
)


Trainer._get_train_sampler = lambda _: None  # prevent shuffling the dataset again

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokenized,
eval_dataset=val_tokenized,
    #compute_metrics=compute_metrics,
#tokenizer=tokenizer,
)

start = time.time()
trainer.train()
#trainer.train(resume_from_checkpoint=True)
print(f"Trained trained in {(time.time() - start)/60} min.")

trainer.save_model(str(MODEL_DIR))
TRAINER_HISTORY_PATH.write_text(json.dumps(trainer.state.log_history))

save_list_to_txt([RANDOM_SEED,
NUM_TOKENIZER_TRAINING_ITEMS,
VOCAB_SIZE,
MODEL_MAX_SEQ_LEN, 
DEVICE_BATCH_SIZE,
gradient_accumulation_steps,
batch_size, 
NUM_EPOCHS,
len(train),
training_args], header+"_tech_info.txt")

save_list_to_txt([("RANDOM_SEED ", RANDOM_SEED),
("NUM_TOKENIZER_TRAINING_ITEMS ", NUM_TOKENIZER_TRAINING_ITEMS),
("VOCAB_SIZE ", VOCAB_SIZE),
("MODEL_MAX_SEQ_LEN ", MODEL_MAX_SEQ_LEN), 
("MODEL_MAX_SEQ_LEN ", DEVICE_BATCH_SIZE),
("gradient_accumulation_steps ", gradient_accumulation_steps),
("batch_size ", batch_size), 
("NUM_EPOCHS ", NUM_EPOCHS),
("len(train)", len(train)),
"training_args", training_args], str(RUN_DIR) + "/" +header+"_tech_info.txt")

