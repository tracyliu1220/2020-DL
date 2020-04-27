from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from loader import *


# random.seed(1)


"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to
implement by yourself.

1. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
2. Output your results (BLEU-4 score, correction words)
3. Plot loss/score
4. Load/save weights
========================================================================================"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
SOS_token = 0
EOS_token = 1
UNK_token = 29
#----------Hyper Parameters----------#
hidden_size = 256
#The number of vocabulary
vocab_size = 30
teacher_forcing_ratio = 0.5
LR = 0.01
MAX_LENGTH = 30
print_every = 5000
target_bleu = 0.00


################################
#Example inputs of compute_bleu
################################
#The target word
reference = 'variable'
#The word generated by your model
output = 'varable'

#compute BLEU-4 score
# def compute_bleu(output, reference):
#    cc = SmoothingFunction()
#     return sentence_bleu([reference], output,weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=cc.method1)

def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # input = input.view(-1)
        # print(input.shape)
        # print(input)
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        # print("\033[38;5;011mdecode forward\033[0m")
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(iter, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    global print_every
    encoder.train()
    decoder.train()

    encoder_hidden = (encoder.initHidden(), encoder.initHidden())

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    #----------sequence to sequence part for encoder----------#
    # encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    # print(input_tensor[0])
    # print(input_tensor[0].size())
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # encoder_outputs[ei] = encoder_output[0, 0]

    # print("----encode finished----")
    # print(encoder_output)


    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    decoder_outputs = ''


    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            topv, topi = decoder_output.topk(1)
            topi = topi.squeeze().detach()
            decoder_outputs += indexToChar(topi)
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            decoder_outputs += indexToChar(decoder_input)

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    if iter % print_every == 0:
        print('>', decoder_outputs)
    
    # print("----decode finished----")

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def test(input_vocab, target_vocab, encoder, decoder, prnt=False):
    encoder.eval()
    decoder.eval()
    global print_output
    
    with torch.no_grad():
        input_tensor = stringToTorch(input_vocab, is_tar=True).to(device)
        target_tensor = stringToTorch(target_vocab, is_tar=True).to(device)

        encoder_hidden = (encoder.initHidden(), encoder.initHidden())

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoder_outputs = ''

        for di in range(25):
            decoder_output, decoder_hidden, = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            decoder_outputs += indexToChar(decoder_input)

            if decoder_input.item() == EOS_token:
                break

        if prnt:
            print('---')
            print('<', input_vocab)
            print('=', target_vocab)
            print('>', decoder_outputs)

    return compute_bleu(decoder_outputs, target_vocab)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    global target_bleu
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # your own dataloader
    # training_pairs = ...
    trainloader = DataSet('train')
    testloader = DataSet('test')

    criterion = nn.CrossEntropyLoss()

    # for iter in range(1, n_iters + 1):
    #     training_pair = training_pairs[iter - 1]
    #     input_tensor = training_pair[0]
    #     target_tensor = training_pair[1]

    # results = []
    with open('results/results.json', 'r') as f:
        results = json.load(f)

    cnt = 0
    for iter in range(1, n_iters + 1):
        idx = random.randint(0, len(trainloader))
        input_vocab, target_vocab = trainloader[idx]

        cnt += 1

        input_tensor = stringToTorch(input_vocab, is_tar=True).to(device)
        target_tensor = stringToTorch(target_vocab, is_tar=True).to(device)

        # print(input_vocab)

        # input_tensor[random.randint(0, len(input_vocab) - 1)][0] = UNK_token

        if iter % print_every == 0:
            print('<', input_vocab)
            print('=', target_vocab)

        loss = train(iter, input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            trainloader = DataSet('train')
            print_loss_avg = print_loss_total / cnt
            print_loss_total = 0
            cnt = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            bleu_score = 0

            print_idx = int(random.random() * len(testloader))
            
            for i in range(len(testloader)):
                prnt = True if i == print_idx else False
                input_vocab, target_vocab = testloader[i]
                bleu_score += test(input_vocab, target_vocab, encoder, decoder, prnt=prnt)

            bleu_score /= len(testloader)
            results.append(bleu_score * 100)
            print('\033[38;5;011mbleu_score: ', bleu_score, '\033[0m')
            if bleu_score >= target_bleu:
                target_bleu = bleu_score
                torch.save(encoder, 'results/weights/v3/{:.2f}-encoder.pth'.format(bleu_score * 100))
                torch.save(decoder, 'results/weights/v3/{:.2f}-decoder.pth'.format(bleu_score * 100))
            with open('results/results.json', 'w') as f:
                f.write(json.dumps(results))
    return results

def demo():
    encoder = torch.load('results/weights/76.14-encoder.pth')
    decoder = torch.load('results/weights/76.14-decoder.pth')
    testloader = DataSet('test2')
    bleu_score = 0
    for i in range(len(testloader)):
        input_vocab, target_vocab = testloader[i]
        bleu_score += test(input_vocab, target_vocab, encoder, decoder, prnt=True)
    bleu_score /= len(testloader)
    print('\033[38;5;011mbleu_score: ', bleu_score, '\033[0m')


encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)
# encoder1 = torch.load('results/weights/76.14-encoder.pth').to(device)
# decoder1 = torch.load('results/weights/76.14-decoder.pth').to(device)
results = trainIters(encoder1, decoder1, 1000000, print_every=print_every) #5000)
# print(results)
# demo()
