"""
We find all similar forms of words in the dictionary, specified by the list. This is necessary to search the text only for those forms represented by a single token.
"""
from pathlib import Path
import pandas as pd
import time

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
)

from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pymorphy3
morph = pymorphy3.MorphAnalyzer()
def find_word_forms(text, word):
    """
    Находит все формы слова `word` в тексте `text`.
    """
    #possitions = []
    #word_forms = []
    possitions_and_word_forms = []
    words = text.lower().split() 
    num = 0
    for text_word in words:
        parsed_word = morph.parse(text_word)[0]  
        if parsed_word.normal_form == word.lower():
            possitions_and_word_forms.append([num,text_word]) 
            #word_forms.append(text_word)
            #possitions.append(num)
        num += 1

    #return word_forms, possitions
    return possitions_and_word_forms


def save_list_to_txt(my_list, filename):
  """
  """
  try:
    with open(filename, 'w', encoding='utf-8') as file:
      for item in my_list:
        file.write(str(item) + '\n')  
  except Exception as e:
    print(f"can't write: {e}")


DATA_DIR = Path("/home/sergeev/Datasets")
MAIN_DIR = Path("/home/sergeev/PythonScripts/NLP/Bert/ru_newspappers_2010-2014/")
TOKENIZER_PATH = MAIN_DIR / "tokenizer.json"

tokenizer = BertTokenizerFast.from_pretrained(str(MAIN_DIR))

base_form_list = ["Россия", "РФ", "Америка", "США", "традиция", "разум", "выжить", "самостоятельно", "индивидуальных", "право", "культура", "экономика", "развитие", "разум"]
#base_form_list = ["традиция", "выжить", "самостоятельно", "индивидуальных", "право", "культура", "экономика", "развитие", "разум"] 

fName = MAIN_DIR / 'vocab.txt'
f = open(fName, encoding="utf8")

start = time.time()
vocab_text = ""
for line in f:
    vocab_text += line
#print(vocab_text[0:2000])
print(f" vocab len: {len(vocab_text)}")


res = []
for base_form in base_form_list:  
    set_of_forms = set() 
    print(base_form)     
    pos_and_form = find_word_forms(vocab_text, base_form)
    print(pos_and_form)
    if len(pos_and_form) != 0:
        for p in pos_and_form:
            if len(tokenizer.encode(p[1], add_special_tokens=False)) == 1: 
                set_of_forms.add(p[1])
    res.append(set_of_forms)

print(res)
save_list_to_txt(res, "250722_vocab_forms_10-14.txt")

print(f"execution in {(time.time()-start)/60} min.")
