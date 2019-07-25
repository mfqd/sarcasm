
import pandas as pd
df = pd.read_json("/valohai/inputs/dataset/Sarcasm_Headlines_Dataset.json", lines=True)
df.head()

import argparse

# In[ ]:


df = df.drop(['article_link'], axis=1)
df.head()


# In[ ]:


df['len'] = df['headline'].apply(lambda x: len(x.split(" ")))
df.head()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        '--epochs',
                        type=int,
                        default=5,
                        help='epoch',
                        )
    parser.add_argument(
                    '--batch_size',
                    type=int,
                    default=100,
                    help='Training batch size (larger batches are usually more efficient on GPUs)',
                    )
    flags = parser.parse_args()
    return flags

flags = parse_args()

# In[ ]:


import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential

max_features = 10000
maxlen = 25
embedding_size = 200

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(df['headline']))
X = tokenizer.texts_to_sequences(df['headline'])
X = pad_sequences(X, maxlen = maxlen)
y = df['is_sarcastic']


# In[ ]:


EMBEDDING_FILE = '/valohai/inputs/glove/glove.6B.200d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:


model = Sequential()
model.add(Embedding(max_features, embedding_size, weights = [embedding_matrix]))
model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(40, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = flags.batch_size
epochs = flags.epochs
history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)


# In[ ]:

import os
OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './outputs')
path = os.path.join(OUTPUTS_DIR, 'fig.png')

import matplotlib.pyplot as plt
fig1, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
fig1.suptitle("Performance of Glove Vectors")
ax1.plot(history.history['acc'])
ax1.plot(history.history['val_acc'])
vline_cut = np.where(history.history['val_acc'] == np.max(history.history['val_acc']))[0][0]
ax1.axvline(x=vline_cut, color='k', linestyle='--')
ax1.set_title("Model Accuracy")
ax1.legend(['train', 'test'])

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
vline_cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]
ax2.axvline(x=vline_cut, color='k', linestyle='--')
ax2.set_title("Model Loss")
ax2.legend(['train', 'test'])

plt.savefig(path)



import json
for i in range(epochs):
    print(json.dumps({'step': i, 'accuracy': history.history['acc'][i]}))
