# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from collections import OrderedDict
# from matplotlib import pyplot as plt
# from models import model_cosine as Model
# from dataset import dataset_dont_train as Data
# from itertools import combinations
# import os
# import numpy as np

# def cosine_similarity(num_epoch):
#     combinations_idx = list(combinations(range(20), 2))
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     dataset = Data.MyDataset()
#     print(dataset)
#     model=Model.SentenceBertJapanese('sonoisa/sentence-bert-base-ja-mean-tokens-v2')
    
#     with tqdm(range(num_epoch)) as epoch_bar:
#         for epoch in epoch_bar:
#             epoch_bar.set_description("[Epoch %d]" % (epoch))
            
#             data_loader = DataLoader(dataset,batch_size=1,shuffle=False, drop_last=True)
#             result=[]
#             count=0
            
#             with tqdm(enumerate(data_loader),
#                       total=len(data_loader),
#                       leave=False) as batch_bar:
#                 for i, (batch) in batch_bar:
#                     # print(batch)
#                     # batch = list(batch)#タプルをリストに
#                     #print(batch)
#                     output = model.encode(batch)#順伝搬
#                     # print(output)
#                     Sum=0
#                     for i in combinations_idx:
#                         cos_sim=torch.cosine_similarity(output[i[0]],output[i[1]],dim=0)
#                         Sum+=cos_sim
#                         # print(cos_sim)
#                     Sum=Sum/20
#                     result.append((cos_sim,count))
#                     count+=1
#         print(result)

# if __name__ == "__main__":
#     cosine_similarity(1)

# from transformers import BertJapaneseTokenizer, BertModel
# import torch





# MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"  # <- v2です。
# model = SentenceBertJapanese(MODEL_NAME)

# sentences = ["暴走したAI"]
# sentence_embeddings = model.encode(sentences, batch_size=8)

# print("Sentence embeddings:", sentence_embeddings)

# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from collections import OrderedDict
# from matplotlib import pyplot as plt
# from models import model_cosine as Model
# from dataset import dataset_dont_train as Data
# from itertools import combinations
# import os
# import numpy as np

# def cosine_similarity(num_epoch):
#     combinations_idx = list(combinations(range(20), 2))
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     dataset = Data.MyDataset()
#     print(dataset)
#     model=Model.SentenceBertJapanese('sonoisa/sentence-bert-base-ja-mean-tokens-v2')

#     with tqdm(range(num_epoch)) as epoch_bar:
#         for epoch in epoch_bar:
#             epoch_bar.set_description("[Epoch %d]" % (epoch))
            
#             data_loader = DataLoader(dataset,batch_size=1,shuffle=False, drop_last=True)
#             result=[]
#             cos=[]
#             count=0
            
#             with tqdm(enumerate(data_loader),
#                       total=len(data_loader),
#                       leave=False) as batch_bar:
#                 for i, (batch) in batch_bar:
#                     # print(batch)
#                     # batch = list(batch)#タプルをリストに
#                     #print(batch)
#                     output = model.encode(batch)#順伝搬
#                     # print(output)
#                     cos.append(output)
#                     if len(cos)==40:
#                         # cos=torch.tensor(cos)
#                         Sum=0
#                         for i in combinations_idx:
#                             # print(i)
#                             # print(cos[i[0]])
#                             # print(cos[i[1]])
#                             cos_sim=torch.cosine_similarity(cos[i[0]][0],cos[i[1]][0],dim=0)
#                             Sum+=cos_sim.item()
#                         for i in cos:
#                             del i
#                         Sum=Sum/190
#                         result.append((Sum,count))
#                         count+=1
#                         cos=[]
#         result=sorted(result,reverse=True)
#         print(result)

# if __name__ == "__main__":
#     cosine_similarity(1)


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

def cosine_similarity(num_epoch):
    combinations_idx = list(combinations(range(20), 2))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Data.MyDataset()
    print(dataset)
    model=Model.SentenceBertJapanese('sonoisa/sentence-bert-base-ja-mean-tokens-v2')

    with tqdm(range(num_epoch)) as epoch_bar:
        for epoch in epoch_bar:
            epoch_bar.set_description("[Epoch %d]" % (epoch))
            
            data_loader = DataLoader(dataset,batch_size=1,shuffle=False, drop_last=True)
            result=[]
            cos=[]
            count=0
            
            with tqdm(enumerate(data_loader),
                      
                      total=len(data_loader),
                      leave=False) as batch_bar:
                
                for i, (batch) in batch_bar:

                    output = model.encode(batch)#順伝搬
                    cos.append(output)

                    if len(cos)==20:
                        Sum=0
                        for i in combinations_idx:
                            cos_sim=torch.cosine_similarity(cos[i[0]][0],cos[i[1]][0],dim=0)
                            # print(cos_sim)
                            # exit()
                            Sum+=cos_sim.item()
                        for i in cos:
                            del i
                        Sum=Sum/(190)
                        result.append((Sum,count))
                        count+=1
                        cos=[]
        result=sorted(result,reverse=True)
        print(result)

if __name__ == "__main__":
    cosine_similarity(1)