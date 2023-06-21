import torch
import pandas as pd
import torch.nn.functional as F

def clickbait_generator(length=5, i=1):
    dt=pd.read_csv('py_model/train1.csv').headline
    dt=dt[:2000]

# create the vocabulary of words to train on
    s1=set()
    for s in dt:
        for w in s.split():
            s1.add(w)
    s1=sorted(s1)
    stoi={}
    cnt=1
    for w in s1:
        stoi.update({w:cnt})
        cnt+=1
        stoi.update({'.':0})

    W = torch.load('py_model/tensor_params2.pth')
    g = torch.Generator().manual_seed(i)
    itos = {value: key for key, value in stoi.items()}
    for i in range(1):
        out = []
        ix = 0
        for i in range(length):
            xenc = F.one_hot(torch.tensor([ix]), num_classes=len(stoi)).float()
            logits = xenc @ W # predict log-counts
            counts = logits.exp() # counts, equivalent to N
            p = counts / counts.sum(1, keepdims=True) # probabilities for next character
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
    return (' '.join(out))