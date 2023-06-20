import pandas as pd
import torch
import torch.nn.functional as F

dt=pd.read_csv('py_model/train1.csv').headline
dt=dt[:2000]

# create the vocabulary of words to train on
s1=set()
for s in dt:
  for w in s.split():
    s1.add(w)

stoi={}
cnt=1
for w in s1:
  stoi.update({w:cnt})
  cnt+=1
stoi.update({'.':0})
len(stoi)

# create the dataset and convert it to tensors
xs, ys = [], []
for w in dt:
  chs = ['.'] + w.split() + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.load('py_model/tensor_params.pth')

for k in range(100):
  
  # forward pass
  xenc = F.one_hot(xs, num_classes=len(stoi)).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  print(loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  # update
  W.data += -600 * W.grad

  torch.save(W, 'py_model/tensor_params.pth')