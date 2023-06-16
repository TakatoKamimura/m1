import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
from matplotlib import pyplot as plt
from models import model_cosine as Model
from dataset import dataset_dont_train as Data
from itertools import combinations
import os
import numpy as np

model=Model.SentenceBertJapanese('sonoisa/sentence-bert-base-ja-mean-tokens-v2')
a='えええええええ'
b='うおおおおおおお'
a1=model.encode(a)
b1=model.encode(b)

print(torch.cosine_similarity(a1[0],b1[0],dim=0))
