import pandas as pd
import torch
import torch.nn.functional as F

dt=pd.read_csv('./train1.csv').headline
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
len(stoi)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create the dataset and convert it to tensors
xs, ys = [], []
for w in dt:
  chs = ['.'] + w.split() + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs).to(device)
ys = torch.tensor(ys).to(device)
num = xs.nelement()
print('number of examples: ', num)
W = torch.load('./tensor_params2.pth').to(device).requires_grad_()
# initialize the 'network'
for k in range(1000):
  # forward pass
  xenc = F.one_hot(xs, num_classes=len(stoi)).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  print(k, loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  W.retain_grad()
  # update
  W.data += -300 * W.grad
torch.save(W, './tensor_params2.pth')