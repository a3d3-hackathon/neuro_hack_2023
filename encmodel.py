from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import copy
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import math as m
from Transformer.preprocess import BASE_DIR
#torch.manual_seed(25)
with open('inputs_pickle', 'rb') as f:
    INPUTS = pickle.load(f)
f.close()

with open('targets_pickle', 'rb') as g:
    TARGETS = pickle.load(g)
g.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########HYPERPARAMETERS###########
batch_size = 512
num_limbs = 5
num_neurons = 1719
d_model = 256
d_ff = 4 * d_model
h = 2
dropout = 0.0
N = 6
lr = 3e-4
max_iters = 100
eval_iters = 10
save_dir = r"C:\Users\jackm\PycharmProjects\Neuro_Hackathon_2023\Output/"
###################################
print(f'CUDA? {torch.cuda.is_available()}')

#base transformer model for neural encoding

def init_normal(sigma):
    def init_(tensor):
        return nn.init.normal_(tensor, mean = 0.0, std=sigma)
    return init_

'''
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    criterion = nn.L1Loss()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits = model(Y)
            loss = criterion()
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
'''

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, bias = True):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias = bias)
        self.w_2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

def attention(query, key, value, dropout=None, attention_mult = 1.0):
    d_k = query.size(-1)

    scores = attention_mult * h * torch.matmul(query, key.transpose(-2, -1)) / d_k
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout, bias = True, attention_mult = 1.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = int(d_model // h)
        self.h = int(h)
        self.linears = clones(nn.Linear(d_model, d_model, bias = bias), 4)
        self.attention_mult = attention_mult
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):

        query, key, value = \
        [l(x).view(-1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self_attn = attention(query, key, value, dropout=self.dropout, attention_mult=self.attention_mult)

        x = x.transpose(1, 2).contiguous().view( -1, self.h * self.d_k)
        x = self.linears[-1](x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer[1](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N, ln_gain_mult):
        super(Decoder, self).__init__()
        self.ln_gain_mult = ln_gain_mult
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size, elementwise_affine=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.ln_gain_mult * self.norm(x)


#############SETUP########################
n = int(0.9*len(INPUTS))
train_inputs = INPUTS[:n]
train_targets = TARGETS[:n]
valid_inputs = INPUTS[n:]
valid_targets = TARGETS[n:]

def get_batch(split):
    #generate batch
    inputs = train_inputs if split == 'train' else valid_inputs
    targets = train_targets if split == 'train' else valid_targets
    if split == 'train':
        ix = torch.randint(len(train_inputs), (batch_size,))
    else:
        ix = torch.randint(len(valid_inputs), (batch_size,))
    x = torch.stack([inputs[i] for i in ix])
    y = torch.stack([targets[i] for i in ix])
    y = y.view(batch_size, 1)
    x.requires_grad = True
    y.requires_grad = True
    x, y = x.to(device), y.to(device)
    return x, y


######test###################


class EncModel(nn.Module):
    def __init__(self, input_size = 1, target_size = num_neurons, N = N, d_model = d_model, d_ff = d_ff,
                 h = h, dropout = dropout, bias = True, attention_mult = 1.0):
        super(EncModel, self).__init__()
        c = copy.deepcopy
        self.d_model = d_model
        self.bias = bias
        attn = MultiHeadAttention(h, d_model, attention_mult)
        ff = FeedForward(d_model, d_ff, dropout)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), N, ln_gain_mult=1.0)

        self.input_positional_embedding = nn.Linear(input_size, d_model)

        self.reco_layer = nn.Linear(d_model, target_size)

    def forward(self, x):
        decoder_input = self.input_positional_embedding(x)
        decoder_output = self.decoder(decoder_input)
        output = self.reco_layer(decoder_output)
        return output



model = EncModel().to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {pytorch_total_params}')


optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.L1Loss()
##########TRAINING LOOP############
epochs = []
train_losses = []
valid_losses = []
#xb: [B, 1719]
#yb: [B, 1]

start = timer()
model.train()
for iter in tqdm(range(1 ,max_iters+1)):
    '''
    #evaluate on train and validation sets
    if iter % eval_interval == 0:
        model.eval()
        X, Y = get_batch('valid')
        _, loss = model(Y)
        print(f"Valid loss: {loss}")
    '''

    xb, yb = get_batch('train')
    output = model(yb)
    loss = criterion(output, xb)
    epochs.append(iter)
    train_losses.append(loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_iters == 0:
        model.eval()
        xb, yb = get_batch('valid')
        output = model(yb)
        valid_loss = criterion(output, xb)
        valid_losses.append(valid_loss.item())
        model.train()
end = timer()

total_time = end - start
print(f"Total training time: {total_time / 60.0} minutes")
plt.plot(epochs, train_losses)
plt.xlabel('Steps')
plt.ylabel('L1 Loss')
plt.savefig(save_dir + 'train.png')
plt.cla()
plt.clf()
plt.plot([i for i in range(0,10)], valid_losses)
plt.xlabel('10 step increment')
plt.ylabel('L1 Loss')
plt.savefig(save_dir + 'valid.png')
#############################