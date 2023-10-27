from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

from gensim.models import Word2Vec, KeyedVectors
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
# import vaderSentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Data(Dataset):
    def __init__(self, df, word2index):
        self.word2index = word2index # to index text

        # sort text by length of total words
        df['length'] = df['text'].apply(lambda x: len(x.split()))
        df.sort_values(by='length', inplace=True)

        # prepare data
        self.features_cols = df.columns.drop(['text', 'category', 'length', 'sa'])
        self.target = df['category'].values
        self.text = df['text'].apply(lambda x: x.split()).values
        self.features = df.loc[:, self.features_cols].values

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        sequence = [self.word2index[w] for w in self.text[item] if w in self.word2index]
        features = self.features[item]
        target = self.target[item]
        return {'target': target, 'features': features, 'sequence': sequence}


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers, output_size, p=.5, use_features=False,
                 features_size=None):

        super().__init__()  # This is needed to initialize the nn.Module properly

        # 'use_features' if used, we supposed to give text and feature data to the model to make prediction.
        # create 'use_features' as 'self.use_features' so we can use it in the hole class functions(e.g. 'forward') instead of just in '__init__' attrbute
        self.use_features = use_features

        # 'Embedding' layer works like a lookup table(dictionary). The words are the keys in this table, while the dense word vectors are the values.
        self.Embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM is the Long Short-Term Memory layer that will take the vectors represntations and extract the semantics/hidden_states
        self.LSTM = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, bidirectional=False, batch_first=True, dropout=p)
        # if 'batch_first=True' 'hidden' shape will be (num_layers * num_directions, batch, hidden_size)
        # if 'batch_first=False' 'hidden' shape will be (num_layers * num_directions, seq_len, hidden_size)
        

        # A Dense/Linear layer that learn from last_hidden_state (optional layer; you can remove it)
        self.Linear = nn.Linear(hidden_size, hidden_size)

        # Dropout layer that help regularizing and prevent overfitting (optional layer)
        # It randomly zeroes some of the elements of the input tensor with probability 'p'
        self.Dropout = nn.Dropout(p)

        # Dense/Linear layer to predict the class of the text/sequence
        self.output = nn.Linear(hidden_size, output_size)

        # add these layers if 'use_features'
        if self.use_features:
            # this dense layer will take the features and tune its parameters
            self.f_Linear = nn.Linear(features_size, hidden_size)

            # this dense layer will take the output from 'f_Linear' and predict the class of the text/sequence
            self.f_output = nn.Linear(hidden_size, output_size)

    def forward(self, x, features=None):  # run the network

        Embedding = self.Embedding(x)  # map the words to their vectors representations by the Embedding layer

        output, (hidden, cell) = self.LSTM(Embedding)  # calculate the sequence/text sementics
        # 'output' comprises all the hidden states in the last layer
        # (hidden, cell) comprises the hidden states after the last timestep

        # 'cell' state contains info about wether to keep a hidden state or no (num_layers * num_directions, batch, hidden_size)
        # 'output' state is a tensor of all the 'hidden' states from each time step  (seq_len, batch, num_directions * hidden_size)
        # 'hidden' state is the hidden states from the last time step  (num_layers * num_directions, batch, hidden_size) 

        last_hidden_state = hidden[-1, :, :]  # last hidden state from the last time step (seq_len, hidden_size)

        x = nn.functional.leaky_relu(self.Linear(last_hidden_state), .001)
        # 'leaky_relu' is similar to relu activation function, it just let the values to be between '0.001' and any other number

        x = self.Dropout(x)  # apply dropout

        # combine the outputs of text and features if 'use_features' is 'True'.
        # we will multiply each output by 0.5 to get half the original weights, and then combine the two halves
        # to have a predictions from both outputs weights.
        if self.use_features:

            x = self.output(x) * 0.5  # half the weights

            x_2 = nn.functional.leaky_relu(self.f_Linear(features), .001)
            x_2 = self.f_output(x_2) * 0.5  # half the weights

            final_x = x + x_2  # combine the two halves outputs

            return final_x

        else:
            return self.output(x)  # if 'use_features' is 'False' then 'features' output won't be calculated


class CNN_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_filters, filter_sizes, output_size, p=.5, use_features=False,
                 features_size=None):
        super().__init__()
        self.use_features = use_features
        self.Embedding = nn.Embedding(vocab_size, embedding_dim)

        # we will build 'Conv2d' layers as much as the amount of 'filter_sizes' list
        # the 'kernel_size' will be (filter_size, embedding_dim),
        # 'filter_size' is the number words to take as 'n-gram'
        # 'embedding_dim' is the vector representation for the word (Embedding dimentions)
        # 'ModuleList' helps in creating more then one layer at once
        #print(filter_sizes)
        self.convs = nn.ModuleList(nn.Conv2d(1, n_filters, (ks, embedding_dim)) for ks in filter_sizes)  # 'convs' will be as much as 'filter_sizes' list

        # note: There is no hidden layer between 'convs' layers and 'output' layer


        # we will concatenate all 'convs' layers outputs and feed them to the 'output' layer
        self.output = nn.Linear(len(filter_sizes) * n_filters, output_size)

        self.Dropout = nn.Dropout(p)

        if self.use_features:
            self.f_Linear = nn.Linear(features_size, hidden_size)
            self.f_output = nn.Linear(hidden_size, output_size)

    def forward(self, x, features=None):
        # the 'Conv2d layer' wants inputs of shape (batch, Channel, Height, Width)
        # our text is of shape(batch, seq_lens)

        # after feeding text to the 'Embedding' layer, we will have shape(batch, seq_lens, embedding_dim)
        Embedding = self.Embedding(x)

        # the convolutional layer wants the inputs to be of
        # shape(batch,  Channel, Height, Width) and the 'Embedding' layer's output is shape(batch, seq_lens, embedding_dim)
        # to make 'Embedding' output correct to feed a 'convs' layer, its shape must be same as the shape of a conv layer.

        # the batch in conv shape exist in 'Embedding' shape.
        # the Channel in conv shape will be the number of embedding layers; in our case it's 1 'Embedding' layer.
        # the Height in conv shape  will be the 'seq_lens'(num words or n-gram)
        # the Width in conv shape  will be the 'embedding_dim'(vector representation of a word)

        # conv(batch, Channel, Height, Width) = Embedding(batch, num_embeddings, seq_lens, embedding_dim)
        Embedding = Embedding.unsqueeze(1) # create Channel dimension

        convs = [F.relu(c(Embedding)).squeeze(3) for c in self.convs]
        # after applying a conv layer, the output will be of shape (batch, Height, Width, Channel)
        # the 'max_pool1d' layer doesn't want the channel dimension(third dim), so we will remove/squeeze it by using .squeeze(3)

        convs = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in convs]
        # the kernel_size is the word embedding_dim (c.shape[2]; which is the Width dimension(second dim) ).
        # since the Width dimension (c.shape[2]) is pooled, the second dimension is empty.
        # so we will remove/squeeze it by using .squeeze(2)

        # we will concatenate all 'convs' outputs (note that 'convs' is a list of all 'convs' layers
        # outputs; e.g. [conv1, conv2, ...])

        # the shape is, 'batch_size' and the sum of all the filters of all 'convs'
        # layers (e.g. n_filter1 + n_filter2 ..., depending on how many conv layer there)
        total_pools = torch.cat(convs, dim=1)  # (batch, total_filters)

        x = self.Dropout(total_pools)

        if self.use_features:
            x = self.output(x) * 0.5
            x_2 = nn.functional.leaky_relu(self.f_Linear(features), .001)
            x_2 = self.f_output(x_2) * 0.5
            final_x = x + x_2
            return final_x
        else:
            return self.output(x)
        
def cnn_padding_to_match_filtersize(text, filter_sizes):
    features_ = []
    for f in text:
        f = f.cpu().numpy().tolist()
        if len(f) < max(filter_sizes):
            f += [1] * (max(filter_sizes) - len(f))
            features_.append(f)
        else:
            features_.append(f)

    return torch.LongTensor(features_).to(device)


def Bert_data(df):
    # sort text by length of words
    df['length'] = df['text'].apply(lambda x: len(x.split()))
    df.sort_values(by='length', inplace=True)

     # prepare data
    features_cols = df.columns.drop(['text', 'category', 'length', 'sa'])
    target = df['category'].values
    text = df['text'] # the BertTokenizer will index them so we don't need for word2index
    features = df.loc[:, features_cols].values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    x_encoded = tokenizer.batch_encode_plus(text, pad_to_max_length=True)
    
    input_ids = torch.tensor(x_encoded['input_ids'],  device=device, dtype=torch.long)
    attention_mask = torch.tensor(x_encoded["attention_mask"],  device=device, dtype=torch.long) # to mark the text; if it was not a padding index,1 else 0
    #print(f'type of target: {type(target)}')
    target = torch.tensor(target,  device=device, dtype=torch.long)
    features = torch.tensor(features, device=device, dtype=torch.float)
    
    return TensorDataset(input_ids, attention_mask, features, target)
    

def prepare_text_dict(all_words_freq):
    # the frequencies of words in text
    freq = all_words_freq
    clean_words = all_words_freq.index
        
    # vocabs set
    clean_words = set(clean_words)
    # convert words to indexes
    word2index = {w: i for i, w in enumerate(clean_words, 2)}  # start at 2
    word2index['<pad>'] = 1
    word2index['<unk>'] = 0

    # convert indexes to words
    index2word = {i: w for i, w in enumerate(clean_words, 2)}  # start at 2
    index2word[1] = '<pad>'
    index2word[0] = '<unk>'
    
    del freq, clean_words # just to save some RAM memory
    
    return word2index, index2word

def pad_text(list_text, seq_length):
    paded_text = []
    for text in list_text:

        if len(text) == seq_length:
            paded_text.append(text)
        else:
            paded_text.append(text + (seq_length - len(text)) * [1]) # '1' is the index for 'PAD' in vocabs dict

    return paded_text


def collate_fn_padded(batch):
    target = [b['target'] for b in batch]
    features = [b['features'] for b in batch]
    sequence = [b['sequence'] for b in batch]
    # extract max_length
    max_length = max([len(b) for b in sequence])
    
    # pad text
    sequence = pad_text(sequence, max_length)

    target = torch.tensor(target, device=device, dtype=torch.long)
    features = torch.tensor(features, device=device, dtype=torch.float)
    sequence = torch.tensor(sequence, device=device, dtype=torch.long)
    # convert list to torch.tensor
    return {'target': target, 'features': features, 'sequence': sequence}
    
def get_mapping_dict(df, word2index, vector_size=50, use_sentiment = False):
    vocab = word2index
    word2vec_embeddings = {}
    print('\nword2vec embeddings...')
    # train a word2vec model on our text
    all_words_lists = [[w for w in text.split()] for text in df['text'].values]
    model = Word2Vec(sentences=all_words_lists, vector_size=vector_size, window=5, min_count=1, workers=4)
    model.save('word2vec.model')
    # load the model
    model = Word2Vec.load('word2vec.model')
    words = list(model.wv.index_to_key)
    for word in vocab.keys():
        if word in words:
            if use_sentiment:
                #word2vec_embeddings[word] = np.append(model.wv[word], df[df['text'].str.contains(word)]['sa'].mean())
                word2vec_embeddings[word] = model.wv[word]* (1 + df[df['text'].str.contains(word)]['sa'].mean())
            else:
                word2vec_embeddings[word] = model.wv[word]
        else:
            word2vec_embeddings[word] = np.zeros(vector_size)

    missing = len(word2index) - len(word2vec_embeddings)
    print(f'word2vec embeddings found for our vocabs: {len(word2vec_embeddings)} | missing: {missing} ; {missing / len(word2index) * 100:.1f}%')
    return word2vec_embeddings

def create_embedding_matrix(word_index, embedding_dict, dimension):
    embedding_matrix = np.zeros((len(word_index) + 1, dimension)) # (len_vocab, 100)

    for word, index in word_index.items():
        if word in embedding_dict.keys():
            embedding_matrix[index] = embedding_dict[word]

    return embedding_matrix

def train_func(model, optimizer, criterion, iterator, using_features=False):
    # create lists to append the scores of each metric from every batch, so we can take the mean/average of each metric as the final scores
    accuracy_list = []
    f1_list = []
    loss_list = []
    
    model.train() # train mode (has specific impact on some layers,e.g. Dropout layer)
    
    for batch in iterator: # loop over the whole data batch-by-batch
        optimizer.zero_grad()
        text = batch['sequence']
        features = batch['features']
        label = batch['target']
        if type(model).__name__ == 'CNN_Model':  # for CNN training
            # CNN is using filter_sizes(e.g. 3, 4, 5), every conv layer has sepcific kernel size
            # a conv layer kernel size is (kernel_size, embedding_dim), so we must pad short texts to the maximum size of filter_sizes list
            # we will do this using the function we created before (in CNN archeticture).

            filter_sizes = [2, 3, 5, 7]
            if using_features:
                output = model(cnn_padding_to_match_filtersize(text, filter_sizes), features)
            else:
                output = model(cnn_padding_to_match_filtersize(text, filter_sizes))

        else:  # for other models
            if using_features:
                output = model(text, features)
            else:
                output = model(text)
                
        loss = criterion(output, label) # calcualte loss function
        loss.backward() # backprobgate loss function

        # Gradient clipping is a technique that tackles exploding gradients. 
        # The idea of gradient clipping is very simple: If the gradient gets too large, we rescale it to keep it small.
        nn.utils.clip_grad_norm_(model.parameters(), 1) # optinal 
        
        # calculate optimizer
        optimizer.step()

        # take the index class for the maximum prediction probability (e.g. [.2, .3, .5], output.argmax(1) will be index 2 is max arg)
        output = output.argmax(1) # 1 will be the 1-axis/dim
        
        # detach: to exclude elements of computation from gradient calculation(backward)
        # cpu: convert the data from GPU cache to CPU cache
        # detach: convert data from torch tensor to np array so we can use any metric(it's build with numpy API so it doesn't except torch API) 
        output = output.detach().cpu().numpy() 
        label = label.detach().cpu().numpy()

        # append metrices values 
        loss_list.append(loss.detach().cpu().numpy().item()) # item will give you just the loss value
        accuracy_list.append(accuracy_score(label, output)) # calculate accuracy_score
        f1_list.append(f1_score(label, output, average='macro')) # calculate f1_score

    # calculate the mean value of the whole epoch
    loss = np.mean(loss_list).round(4) # round(4) will give us the last 4 numbers after '.' (e.g. 2.9567)
    f1 = np.round(np.mean(f1_list) * 100, 1) # multiply the value with 100 so it will look like percentage (e.g. 0.99 * 100 = 99)
    accuracy = np.round(np.mean(accuracy_list) * 100, 1)  # multiply the value with 100 so it will look like percentage (e.g. 0.99 * 100 = 99)

    return loss, f1, accuracy

def eval_func(model, criterion, iterator, using_features=False):
    # create lists to append the scores of each metric from every batch, so we can take the mean/average of each metric as the final scores
    accuracy_list = []
    f1_list = []
    loss_list = []

    model.eval() # train mode (has specific impact on some layers,e.g. Dropout layer)
    
    for batch in iterator:# loop over the whole data batch-by-batch
        # PREPARE DATA
        text = batch['sequence']
        features = batch['features']
        label = batch['target']
            
        # FEED MODEL
        
        with torch.no_grad(): # don't  calculate  gradients
            if type(model).__name__ == 'CNN_Model':  # for CNN model
                # CNN is using filter_sizes(e.g. 3, 4, 5), every conv layer has sepcific kernel size
                # conv layer kernel size is of shape(kernel_size, embedding_dim), so we must pad short texts to the maximum size filter_sizes
                # we will do this using the function we created before (in CNN archeticture).
                filter_sizes = [2, 3, 5, 7]
                if using_features: 
                    output = model(cnn_padding_to_match_filtersize(text, filter_sizes), features)
                else:
                    output = model(cnn_padding_to_match_filtersize(text, filter_sizes))

            else:  # for other models
                if using_features:
                    output = model(text, features)
                else:
                    output = model(text)

            loss = criterion(output, label) # calcualte loss function

        # take the index class for the maximum prediction probability (e.g. [.2, .3, .5], output.argmax(1) will be index 2 is max arg)
        output = output.argmax(1) # 1 will be the 1-axis/dim
        
        # detach: to exclude elements of computation from gradient calculation(backward)
        # cpu: convert the data from GPU cache to CPU cache
        # detach: convert data from torch tensor to np array so we can use any metric(it's build with numpy API so it doesn't except torch API) 
        output = output.detach().cpu().numpy() 
        label = label.detach().cpu().numpy()

        # append metrices values 
        loss_list.append(loss.detach().cpu().numpy().item()) # item will give you just the loss value
        accuracy_list.append(accuracy_score(label, output)) # calculate accuracy_score
        f1_list.append(f1_score(label, output, average='macro')) # calculate f1_score

    # calculate the mean value of the whole epoch
    loss = np.mean(loss_list).round(4) # round(4) will give us the last 4 numbers after '.' (e.g. 2.9567)
    f1 = np.round(np.mean(f1_list) * 100, 1) # multiply the value with 100 so it will look like percentage (e.g. 0.99 * 100 = 99)
    accuracy = np.round(np.mean(accuracy_list) * 100, 1)  # multiply the value with 100 so it will look like percentage (e.g. 0.99 * 100 = 99)

    return loss, f1, accuracy

# --- pred loop --- #
def pred(model, iterator, loss_function, using_features=False):
    # create lists to append the scores of each metric from every batch, so we can take the mean/average of each metric as the final scores
    accuracy_list = []
    f1_list = []
    loss_list = []

    model.eval() # train mode (has specific impact on some layers,e.g. Dropout layer)
    
    for batch in iterator:# loop over the whole data batch-by-batch
        # PREPARE DATA 
        # extract non-BERT data if the current model is not BERT
        text = batch['sequence']
        features = batch['features']
        label = batch['target']

        # FEED DATA
        with torch.no_grad(): # don't calculate gradients
            if type(model).__name__ == 'CNN_Model':  # for CNN model
                # CNN is using filter_sizes(e.g. 3, 4, 5), every conv layer has sepcific kernel size
                # conv layer kernel size is of shape(kernel_size, embedding_dim), so we must pad short texts to the maximum size filter_sizes
                # we will do this using the function we created before (in CNN archeticture).
                
                filter_sizes = [2, 3, 5, 7]
                if using_features:
                    output = model(cnn_padding_to_match_filtersize(text, filter_sizes), features)
                else:
                    output = model(cnn_padding_to_match_filtersize(text, filter_sizes))

            else:  # for other models
                if using_features:
                    output = model(text, features)
                else:
                    output = model(text)

            loss = loss_function(output, label) # calcualte loss function
        
        # take the index class for the maximum prediction probability (e.g. [.2, .3, .5], output.argmax(1) will be index 2 is max arg)
        output = output.argmax(1) # 1 will be the 1-axis/dim
        
        # detach: to exclude elements of computation from gradient calculation(backward)
        # cpu: convert the data from GPU cache to CPU cache
        # detach: convert data from torch tensor to np array so we can use any metric(it's build with numpy API so it doesn't except torch API) 
        output = output.detach().cpu().numpy() 
        label = label.detach().cpu().numpy()

        # append metrices values 
        loss_list.append(loss.detach().cpu().numpy().item()) # item will give you just the loss value
        accuracy_list.append(accuracy_score(label, output)) # calculate accuracy_score
        f1_list.append(f1_score(label, output, average='macro')) # calculate f1_score

    # calculate the mean value of the whole epoch
    loss = np.mean(loss_list).round(4) # round(4) will give us the last 4 numbers after '.' (e.g. 2.9567)
    f1 = np.round(np.mean(f1_list) * 100, 1) # multiply the value with 100 so it will look like percentage (e.g. 0.99 * 100 = 99)
    accuracy = np.round(np.mean(accuracy_list) * 100, 1)  # multiply the value with 100 so it will look like percentage (e.g. 0.99 * 100 = 99)
    name = type(model).__name__ # the name of the Model

    return loss, f1, accuracy, name

def training_model(model, optimizer, criterion, train_iterator, eval_iterator, epochs=1000, using_features=False):
    print('\nTraining Started...\n')
    # for early stopping
    stop = 0  # increase this number if the model didn't become more accurate compared to the last best epoch
    min_eval_acc = 0. # assign this number as the best evaluation accuracy score
    best_epoch = 0 # assign this number as the best epoch
    best_loss = 0. # assign this number as the best loss
    best_acc = 0. # assign this number as the best accuracy
    best_f1 = 0. # assign this number as the best f1-score
    for epoch in range(epochs): # Epochs Loop 
    
        loss, f1, acc = train_func(model, optimizer, criterion, train_iterator, using_features)  # train model
        eval_loss, eval_f1, eval_acc = eval_func(model, criterion, eval_iterator, using_features)  # evaluate model


        # save current-epoch model file, so we can coninue training later (fine-tuning the paremeters of last epoch). and delete
        # the previous current-epoch model file, to avoid making lots of files(you can delete this section if you want)
        # best model scores
        if eval_acc > min_eval_acc: # current accuracy greater then the best accuracy 
            min_eval_acc = eval_acc # assign 'min_eval_acc' as the best accuracy
            #training scores
            best_loss = loss
            best_acc = acc
            best_f1 = f1
            
            #evaluation scores
            best_eval_loss = eval_loss
            best_eval_acc = eval_acc
            best_eval_f1 = eval_f1

            best_model = model.state_dict() # best model parameters
            best_epoch = epoch + 1  # '+ 1' because 'epoch' starts from 0
            stop = 0 # reset 'stop' counter
            # print the current as the 'Best Epoch score'
            print(
                f'BEST Epoch({best_epoch} | Train(loss: {best_loss} | acc: {best_acc} | f1-score: {best_f1}) & Eval(loss: {best_eval_loss} | acc: {best_eval_acc} | f1-score: {best_eval_f1})\n')
            #return best_model, best_epoch, best_loss, best_acc, best_f1, best_eval_loss, best_eval_acc, best_eval_f1

        # if current accuracy wasn't greater then the best accuracy add 1 to 'stop'
        else:
            stop += 1
            # print the current epoch as the 'Normal Epoch score'
            print(
                f'Epoch({epoch + 1} | Train(loss: {loss} | acc: {acc} | f1-score: {f1}) & Eval(loss: {eval_loss} | acc: {eval_acc} | f1-score: {eval_f1})\n')
        
        # if 'stop' reached 5 it will stop training
        if stop == 10:
            print('EARLY STOPPING!')
            # print the best epoch recorded during the whole training
            print(
                f'BEST EPOCH({best_epoch} | Train(loss:{best_loss} | acc:{best_acc} | f1-score:{best_f1}) & Eval(loss: {best_eval_loss} | acc: {best_eval_acc} | f1-score: {best_eval_f1})\n')
            #return [best_model, best_epoch, best_loss, best_acc, best_f1, best_eval_loss, best_eval_acc, best_eval_f1]
            # save best model
            # torch.save(best_model, f'epoch_{best_epoch}_val_loss_{best_eval_loss}_val_acc_{best_eval_acc}_val_f1_{best_eval_f1}.pt')
            break