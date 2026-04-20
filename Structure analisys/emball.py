# -*- coding: utf-8 -*-
"""
We generate a database (ChromaDB) of embeddings for the selected forms.
"""

from pathlib import Path
import time
import datasets
import pandas as pd
import torch

from tokenizers import BertWordPieceTokenizer, Regex, normalizers
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
)
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import numpy as np
import chromadb

print(torch.__version__)
print(torch.cuda.is_available())

def emb_sum_4last(hidden_states):
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)
    # Sum the vectors from the last four layers.
    token_vecs_sum = []

    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0) # Sum the vectors from the last four layers. 
        token_vecs_sum.append(sum_vec)
    return token_vecs_sum

def get_emb_cont(text, form):
  """
  Usable for compound embeddins 
  """
  if len(tokenizer.encode(form, add_special_tokens=False)) == 1:
      input_ids_form = tokenizer.encode(form, add_special_tokens=False)
      input_ids_snt = tokenizer.encode(text, add_special_tokens=False)
      #position = input_ids_snt.index(input_ids_form[0])
      position = -1
      for i in range(len(input_ids_snt) - len(input_ids_form) + 1):
          if input_ids_snt[i:i + len(input_ids_form)] == input_ids_form:
              position = i
              break
  
      if position == -1:
          print(f"Форма '{form}' не найдена в предложении '{text}'.  Возвращаю np.nan.")
          return np.nan
      input_ids_snt_tensor = torch.tensor(input_ids_snt).unsqueeze(0)
      hidden_states_1 = model(input_ids_snt_tensor).hidden_states
      hidden_states_1 = emb_sum_4last(hidden_states_1)
      return hidden_states_1[position].detach().numpy().reshape(1, -1)
  else: return [np.nan]
  
def limit_rows_per_value(df, column_name, max_rows_per_value=10):
    """
    Оставляет в DataFrame не более max_rows_per_value строк для каждого
    уникального значения в указанной колонке.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        column_name (str): Имя колонки, по которой производится фильтрация.
        max_rows_per_value (int): Максимальное количество строк, которое нужно
                                   оставить для каждого уникального значения.

    Returns:
        pd.DataFrame: DataFrame, отфильтрованный по указанному критерию.
    """

    filtered_df = pd.DataFrame()  # Создаем пустой DataFrame для результата

    for value in df[column_name].unique():
        # Получаем все строки, где значение в колонке column_name равно value
        subset = df[df[column_name] == value]

        # Ограничиваем количество строк до max_rows_per_value
        subset = subset.head(max_rows_per_value)

        # Добавляем подмножество к отфильтрованному DataFrame
        filtered_df = pd.concat([filtered_df, subset], ignore_index=True)

    return filtered_df

DATA_DIR = Path("/home/sergeev/Datasets")
#MAIN_DIR = Path("/home/sergeev/PythonScripts/NLP/Bert/ru_newspappers_99-2009")
#MODEL_DIR = MAIN_DIR / "run_20250625-2247/model"
MAIN_DIR = Path("/home/sergeev/PythonScripts/NLP/Bert/ru_newspappers_2010-2014")
#MODEL_DIR = MAIN_DIR / "run_20250718-2149/training_checkpoints/checkpoint-50000"
MODEL_DIR = MAIN_DIR / "run_20250718-2149/model"
TOKENIZER_PATH = MAIN_DIR / "tokenizer.json"
#df_st = pd.DataFrame({"параметры": ["mean", "min", "max", "var", "ex_num"]})

model = BertForMaskedLM.from_pretrained(str(MODEL_DIR), output_hidden_states = True)
tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
model.eval() 

#ds = load_dataset("csv", data_files= str(DATA_DIR) + '//RR_US_wsf_ru_newspappers_99-2009_512seqlen.csv')
#ds = load_dataset("csv", data_files= str(DATA_DIR) + '//inglt_se_ru_newspappers_99-2009_512seqlen.csv')
ds = load_dataset("csv", data_files= str(DATA_DIR) + '//250722_s_ru_newspappers_2010-2014_512seqlen.csv')
ds = ds['train']
#ds=ds[len(ds)-5:len(ds)]
#ds = ds[0:10]


df = pd.DataFrame(ds)
df['Индекс']=df.index
print(df)
print(f"len of data: {len(df)}")
#df = df[['Индекс', "Базовая форма", "Номер примера", "Номер слова", "Предложение", "Искомая форма"]]
df = df[['Индекс', "Базовая форма", "Предложение", "Искомая форма"]]

df["Базовая форма"] = df["Базовая форма"].str.replace("росс", "Россия")
df["Базовая форма"] = df["Базовая форма"].str.replace("рф", "РФ")
df["Базовая форма"] = df["Базовая форма"].str.replace("амер", "Америка")
df["Базовая форма"] = df["Базовая форма"].str.replace("сша", "США")
df["Базовая форма"] = df["Базовая форма"].str.replace("трад", "традиция")
df["Базовая форма"] = df["Базовая форма"].str.replace("разу", "разум")
df["Базовая форма"] = df["Базовая форма"].str.replace("выжи", "выжить")
df["Базовая форма"] = df["Базовая форма"].str.replace("само", "самостоятельно")
df["Базовая форма"] = df["Базовая форма"].str.replace("инди", "индивидуальных")
df["Базовая форма"] = df["Базовая форма"].str.replace("прав", "право")
df["Базовая форма"] = df["Базовая форма"].str.replace("куль", "культура")
df["Базовая форма"] = df["Базовая форма"].str.replace("экон", "экономика")
df["Базовая форма"] = df["Базовая форма"].str.replace("разв", "развитие")


#df = limit_rows_per_value(df, "Базовая форма", 10000) # ограничиваем вхождение каждой формы в датасет 10к экземплярами

start = time.time()
df['Эмбеддинг'] = df.apply(lambda x: get_emb_cont(x['Предложение'], x['Искомая форма']), axis = 1)
df=df.dropna()
df['Эмбеддинг'] = df.apply(lambda x: x['Эмбеддинг'][0], axis=1) # can use df.dropna() for del nan

#df['Эмбеддинг'] = df.apply(lambda x: x['Эмбеддинг'], axis=1)
print(f"finished in {time.time()-start} seconds")
print(df)

#client = chromadb.PersistentClient(path="/home/sergeev/PythonScripts/NLP/Bert/ru_newspappers_99-2009//ChromaDB_store_99")
client = chromadb.PersistentClient(path="/home/sergeev/PythonScripts/NLP/Bert/ru_newspappers_2010-2014//ChromaDB_store_10-14")

#collection_name = "RU99-2009_2" 
#collection_name = "RNP_run_20250718_model_lim10k"
collection_name = "RNP_run_20250718_model"

try:
  #collection = client.get_collection(name="collection_name")
  #collection.delete(ids=df['Индекс'].astype(str).tolist())
  client.delete_collection(collection_name)
except:
  print(f"collection with name {collection_name} deleted")

collection = client.create_collection(name=collection_name)
print(f"Collection '{collection_name}' created.")

from chromadb.api import BaseAPI
maxbatchsize = 5461#get_max_batch_size()

if len(df) > maxbatchsize:
  for i in range(0, len(df), maxbatchsize):
    df_t = df[i:i+maxbatchsize]

    ids = df_t['Индекс'].astype(str).tolist()
    documents = df_t['Предложение'].tolist()  
    embeddings = df_t['Эмбеддинг'].tolist()  
    #metadatas = df_t[['Базовая форма', 'Номер примера', 'Номер слова', 'Искомая форма']].to_dict('records')
    metadatas = df_t[['Базовая форма', 'Искомая форма']].to_dict('records')
    collection.add(
          ids=ids,
          documents=documents, # все еще нужно указать documents, т.к. этого требует API
          embeddings=embeddings,
          metadatas=metadatas
    )
print("collection added")


##print(client.list_collections())
##collection_name = "RU99_test_1"
##client.delete_collection(collection_name)
##print(client.list_collections()) 

#try:
#    collection = client.get_collection(name=collection_name)
#    print(f"Collection '{collection_name}' loaded.")
#except ValueError:
#    print(f"Collection '{collection_name}' not found.  Make sure the database exists and the collection name is correct.")
#    exit() 
#print(f"collection.count() {collection.count()}")

#def split_list(input_list, chunk_size):
#    for i in range(0, len(input_list), chunk_size):
#        yield input_list[i:i + chunk_size]
#        
#split_docs_chunked = split_list(split_docs, 5461)
#
#for split_docs_chunk in split_docs_chunked:
#  collection.add(
#      ids=ids,
#      documents=documents, # все еще нужно указать documents, т.к. этого требует API
#      embeddings=embeddings,
#      metadatas=metadatas
#  )

