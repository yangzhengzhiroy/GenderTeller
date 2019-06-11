import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def encode(name_ls):
    """
    Encode list of names into list of list of character IDs using the character encoder.
    :param name_ls: list of names
    :return: list (each name) of list (each word) of character IDs
    """
    name_id2word_id2char_ids = []
    for name in name_ls:
        name = list(name)
        name_id2word_id2char_ids.append(encoder.transform(['^'] + name + ['$']).tolist())
    return name_id2word_id2char_ids


class KerasBatchGenerator(object):

    def __init__(self, data, label, batch_size):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.current_idx = 0
        self.num_of_batches = len(data) // batch_size

    def generate(self):
        while True:
            if self.current_idx >= self.num_of_batches:
                self.current_idx = 0
            for index in range(self.num_of_batches):
                x, y = self.data[(self.current_idx * self.batch_size):((self.current_idx + 1) * self.batch_size)], \
                       self.label[(self.current_idx * self.batch_size):((self.current_idx + 1) * self.batch_size)]
                self.current_idx += 1
                yield x, y


with open('init_epoch.json', 'r') as f:
    init_epoch = json.load(f)
init_epoch = init_epoch['epoch']
data_df = pd.read_csv('data/final_data.csv')
data_df = data_df.dropna()
encoder = LabelEncoder()
characters = list(set(''.join(data_df['full_name'])))
characters = ['^', '$'] + characters
encoder.fit(characters)
with open('char_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f, protocol=pickle.HIGHEST_PROTOCOL)
print('10%')

index = np.arange(0, len(data_df.index))
np.random.seed(123)
np.random.shuffle(index)
names, genders = data_df['full_name'].values[index], data_df['GenderTeller'].values[index]
names = encode(names)
max_len = 103
names = pad_sequences(names, padding='post')
print(names.shape)
print('20%')

labelencoder = LabelEncoder()
labelencoder.fit(list(set(genders)))
genders = labelencoder.transform(genders)
print('30%')

X_train, X_test, y_train, y_test = train_test_split(names, genders, test_size=0.2, random_state=123)
print(len(X_train), len(X_test))
X_train_gen = KerasBatchGenerator(X_train, y_train, 128)
X_test_gen = KerasBatchGenerator(X_test, y_test, 128)
print('40%')

checkpoint = ModelCheckpoint('gender_model.h5', save_best_only=True)
earlystop = EarlyStopping(patience=2)
history = History()
"""
model = Sequential()
model.add(Embedding(len(characters), output_dim=256))
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(80, 39, 256)))
model.add(Dropout(rate=0.2))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(rate=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print('50%')
"""
model = load_model('gender_model.h5')
model.fit_generator(X_train_gen.generate(), len(X_train) // 128, epochs=init_epoch + 10,
                    validation_data=X_test_gen.generate(), validation_steps=len(X_test) // 128,
                    callbacks=[earlystop, checkpoint, history], initial_epoch=init_epoch)
print('80%')
print(model.history.history)
print(init_epoch)
with open('init_epoch.json', 'w') as f:
    json.dump({'epoch': init_epoch + len(model.history.history['val_loss'])}, f)

y_train_pred = model.predict_classes(X_train)
y_train_pred = labelencoder.inverse_transform(y_train_pred)
y_train_true = labelencoder.inverse_transform(y_train)
print("training accuracy:", accuracy_score(y_train_true, y_train_pred))
print("training confusion:", confusion_matrix(y_train_true, y_train_pred))
y_test_pred = model.predict_classes(X_test)
y_test_pred = labelencoder.inverse_transform(y_test_pred)
y_test_true = labelencoder.inverse_transform(y_test)
print("testing accuracy:", accuracy_score(y_test_true, y_test_pred))
print("training confusion:", confusion_matrix(y_test_true, y_test_pred))
print('100%')
