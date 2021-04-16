# from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig
from transformers import BertForSequenceClassification, AdamW, BertConfig, AutoConfig
from transformers import BertTokenizer
#from keras.preprocessing.sequence import pad_sequences
import operator
import math

import torch
import boto3
import os
import tarfile
import io
import base64
import json
import re

s3 = boto3.client('s3')


class ServerlessModel:
    def __init__(self, model_path=None, s3_bucket=None, file_prefix=None):
        self.model, self.tokenizer = self.from_pretrained(model_path, s3_bucket, file_prefix)

    def from_pretrained(self, model_path: str, s3_bucket: str, file_prefix: str):
        model = self.load_model_from_s3(model_path, s3_bucket, file_prefix)
        tokenizer = self.load_tokenizer(model_path)
        return model, tokenizer

    def load_model_from_s3(self, model_path: str, s3_bucket: str, file_prefix: str):
        if model_path and s3_bucket and file_prefix:
            obj = s3.get_object(Bucket=s3_bucket, Key=file_prefix)
            bytestream = io.BytesIO(obj['Body'].read())
            tar = tarfile.open(fileobj=bytestream, mode="r:gz")
            config = AutoConfig.from_pretrained(f'{model_path}/config.json')
            for member in tar.getmembers():
                if member.name.endswith(".bin"):
                    f = tar.extractfile(member)
                    #state = torch.load(io.BytesIO(f.read()))
                    state = torch.load(io.BytesIO(f.read()), map_location=torch.device('cpu'))
                    # model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=None, state_dict=state, config=config) # Original from example
                    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=None, state_dict=state, config=config)
                    print('model created!')

            return model
        else:
            raise KeyError('No S3 Bucket and Key Prefix provided')

    def load_tokenizer(self, model_path: str):
        #tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return tokenizer

    def encode(self, sentence):
        encoded = self.tokenizer.encode_plus(sentence)
        return encoded["input_ids"], encoded["attention_mask"]

    def decode(self, token):
        answer_tokens = self.tokenizer.convert_ids_to_tokens(token, skip_special_tokens=True)
        return self.tokenizer.convert_tokens_to_string(answer_tokens)

    def predict(self, sentence):
        input_ids, attention_mask = self.encode(sentence)
        # cast to tensor
        input_ids = torch.tensor(input_ids)
        attn_mask = torch.tensor(attention_mask)
        # add an extra dim for the "batch
        input_ids = input_ids.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(0)

        # BERT Model
        # set model in evaluation mode (dropout layers behave differently during evaluation)
        self.model.eval()

        # copy inputs to device
        # input_ids = input_ids.to(device)
        # attn_mask = attn_mask.to(device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, token_type_ids=None, attention_mask=attn_mask)

        logits_ = logits[0].detach().cpu().numpy()[0]
        # print(logits_)
        position_class = [i for i, j in enumerate(logits_) if j == max(logits_)][0]

        # label_set={'BETTER':0,'NONE':1, 'WORSE':2}
        if position_class == 1:
            classification = "subjective"
        else:
            classification = "non_subjective"

        return classification
