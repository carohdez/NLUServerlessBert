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

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

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

        prob_threshold = 0.8
        main_aspect = ""
        max_prob = 0
        aspects = {'facilities': 0, 'staff': 1, 'room': 2, 'bathroom': 3, 'location': 4, 'price': 5, 'ambience': 6,
                   'food': 7, 'comfort': 8, 'checking': 9}

        for i in range(0, 9):
            index, value = max(enumerate(logits_), key=operator.itemgetter(1))
            if index < 9:  # Having issues with position 10, needs to be fixed.
                # if value > 0:
                #     print("Aspect " + str(i) + ": " + list(aspects.keys())[
                #         list(aspects.values()).index(index)] + ", logit: " + str(value) + ", prob: " + str(
                #         format(sigmoid(value), ".2f")))
                prob_value = 1 / (1 + math.exp(-value))
                if ( prob_value >= prob_threshold) & (prob_value > max_prob):
                    main_aspect = list(aspects.keys())[list(aspects.values()).index(index)] # only return aspect with greatest probability
                    max_prob = prob_value
                logits_[index] = 0
        if len(main_aspect) > 0:
            return main_aspect
        else:
            return "none"


