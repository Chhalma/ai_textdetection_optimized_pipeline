##  All imprts and Desklib model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel
#from transformers.modeling_utils import PreTrainedModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
from pathlib import Path
from tqdm import tqdm
from prettytable import PrettyTable
from contextlib import nullcontext
from transformers import BitsAndBytesConfig
from datetime import datetime
import os, gc, json, time, psutil




# 1. ENVIRONMENT VARIABLES FOR MEMORY OPTIMIZATION
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 2. REDUCE TRANSFORMERS CACHE
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface/transformers'


print("🔧 Memory optimization settings applied")

class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        #  Cast pooled_output to same dtype as classifier weights
        pooled_output = pooled_output.to(self.classifier.weight.dtype)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output
