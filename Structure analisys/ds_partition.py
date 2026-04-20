
from pathlib import Path
import pandas as pd
import time
from datasets import load_dataset
import nltk
from transformers import AutoTokenizer, BertTokenizerFast
from tokenizers import Regex, normalizers

def list_into_text(data):
    text = ""
    for i in data:
        text += i + " "
    return text

def list_to_text(data, separator):
    return separator.join(data)
    
def chunk_text_into_sentences_and_tokenize_nltk(text, tokenizer, max_tokens=512):
    sentences = nltk.sent_tokenize(text)  
    chunks = []
    current_chunk = ""
    current_chunk_token_count = 0
    except_cnt = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        sentence_token_count = len(sentence_tokens)

        if sentence_token_count > max_tokens:
            print(f"Warning: Sentence exceeds max_tokens ({max_tokens}). Consider splitting it further: {sentence[0:50]}")
            except_cnt += 1
            continue

        if current_chunk_token_count + sentence_token_count <= max_tokens:
            current_chunk += sentence + " "  
            current_chunk_token_count += sentence_token_count
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_chunk_token_count = sentence_token_count

    if current_chunk:
        chunks.append(current_chunk.strip())
    print(f"There were {except_cnt} exceptions")
    
    return chunks
    
DATA_DIR = Path("/home/sergeev/Datasets/")
ds = load_dataset("csv", data_files= str(DATA_DIR) + '//ru_newspappers_2015-2019.csv')
ds = ds['train']
#ds=ds[0:10]
df = pd.DataFrame(ds)
d_text = list_to_text(df['text'], " ")
print(len(d_text))

#tokenizer_name = "bert-base-uncased" 
TOKENIZER_PATH = "/home/sergeev/PythonScripts/NLP/Bert/ru_newspappers_2015-2019/tokenizer.json"
tokenizer_name = TOKENIZER_PATH
max_tokens = 512
tokenizer = BertTokenizerFast(tokenizer_file=tokenizer_name) # may ask for dir
print(tokenizer)

start = time.time()
chunks = chunk_text_into_sentences_and_tokenize_nltk(d_text, tokenizer, max_tokens)
print(f"Chunks created in {(time.time() - start)/60} min.")


sent_lengths = []
for i, chunk in enumerate(chunks):
    #token_count = len(AutoTokenizer.from_pretrained(tokenizer_name).tokenize(chunk))
    token_count = len(tokenizer.encode(chunk))
    sent_lengths.append(token_count)
    #print(f"Chunk {i+1} (Length: {token_count} tokens):\n{chunk}\n---")
print(f"number of chunks {len(sent_lengths)} \nnumber of tokens {sum(sent_lengths)} \nmax length {max(sent_lengths)}")

df_csv = pd.DataFrame({"text":chunks})
print(df_csv[0:10])
df_csv.to_csv("ru_newspappers_2015-2019_512seqlen.csv")
df_csv[0:1000000].to_csv("ru_newspappers_2015-2019_1M_512seqlen.csv")