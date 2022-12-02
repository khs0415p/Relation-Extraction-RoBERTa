import pickle as pickle
import numpy as np
import os
import pandas as pd
import torch
import tqdm


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

class Preprocess:
  def __init__(self, path):
    self.data = self.load_data(path)
  
  def load_data(self, path):
    data = pd.read_csv(path)
    
    return data
  
  def label_to_num(self, label):
    num_label = []

    with open('./NLP_dataset/dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num= pickle.load(f)
      for val in label:
          num_label.append(dict_label_to_num[val])
        
    return num_label
  
  def tokenized_dataset(self, dataset, tokenizer, multi=False):
    print(dataset['sentence'].iloc[0:10])

    entity_loc_ids = []
    concats = []

    for sent, sub_e, obj_e, s_t, o_t in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity'], dataset['subject_type'], dataset['object_type']):
      temp = f"@*{s_t}*{sub_e}@와 #^{o_t}^{obj_e}#의 관계는 무엇인가요?"
      current_entity_loc_ids = self.make_entity_ids(sentence=sent, tokenizer=tokenizer, multi_sentence=temp, multi=multi)
      entity_loc_ids.append(current_entity_loc_ids)
      concats.append(temp)

    if multi:

      tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        concats,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )

    else:
      tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )

    tokenized_sentences['entity_loc_ids'] = torch.LongTensor(entity_loc_ids)
    
    return tokenized_sentences

  def make_entity_ids(self, sentence, tokenizer, temp, multi):

    entity_loc_ids = [0] * 256

    type_to_num={
        '사람': 1,
        '조직': 2,
        '날짜': 3,
        '장소': 4,
        '단어': 5,
        '숫자': 6,
      }

    if multi:
    # roberta - multi config
      tokenized_sentence = tokenizer.tokenize(sentence, temp, padding="max_length", truncation=True, max_length=256, add_special_tokens=True)
      tokenized_sentence = np.array(tokenized_sentence)
      sub_indices = np.where(tokenized_sentence == '@')[0] # sen 0 , 1 q 2, 3
      sub_type_indices = np.where(tokenized_sentence == '*')[0]
      obj_indices = np.where(tokenized_sentence == '#')[0]
      obj_type_indices = np.where(tokenized_sentence == '^')[0]

      entity_loc_ids[sub_type_indices[1]+1: sub_indices[1]] = [1] * (sub_indices[1] - sub_type_indices[1]-1)
      entity_loc_ids[sub_type_indices[3]+1: sub_indices[3]] = [1] * (sub_indices[3] - sub_type_indices[3]-1)
      entity_loc_ids[obj_type_indices[1]+1: obj_indices[1]] = [2] * (obj_indices[1] - obj_type_indices[1]-1)
      entity_loc_ids[obj_type_indices[3]+1: obj_indices[3]] = [2] * (obj_indices[3] - obj_type_indices[3]-1) 

    else:
      tokenized_sentence = tokenizer.tokenize(sentence, padding="max_length", truncation=True, max_length=256, add_special_tokens=True)
      tokenized_sentence = np.array(tokenized_sentence)
      sub_indices = np.where(tokenized_sentence == '@')[0] #
      sub_type_indices = np.where(tokenized_sentence == '*')[0]
      obj_indices = np.where(tokenized_sentence == '#')[0]
      obj_type_indices = np.where(tokenized_sentence == '^')[0]

      entity_loc_ids[sub_type_indices[-1]+1: sub_indices[-1]] = [1] * (sub_indices[-1] - sub_type_indices[-1]-1)
      entity_loc_ids[obj_type_indices[-1]+1: obj_indices[-1]] = [2] * (obj_indices[-1] - obj_type_indices[-1]-1)

            
    return entity_loc_ids