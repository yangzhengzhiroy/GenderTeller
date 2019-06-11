# -*- coding: UTF-8 -*-
"""
<<<<<<< HEAD
This module prepare the input names and create model object.
"""
import os
import re
import pickle
import logging
import numpy as np
from .utils import log_config, setup_logging
from genderteller import PARENT_DIR, gender_class, gender_cutoff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
=======
This module creates model object.
"""
import os
import logging
import numpy as np
from subprocess import Popen, PIPE
from .utils import log_config, setup_logging
from genderteller import PARENT_DIR, gender_class, gender_cutoff
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from .encoder import GenderEncoder, NameEncoder, KerasBatchGenerator
>>>>>>> 4fe210c4f225ae30b22b0fbeab56621621b768ea


setup_logging(log_config)
logger = logging.getLogger(__name__)


<<<<<<< HEAD
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


class GenderEncoder(object):
    """ Encode the gender to categories. """
    _gender_encoder_file_name = 'gender_encoder.pkl'
    _encoder_path = os.path.join(PARENT_DIR, 'models', _gender_encoder_file_name)

    def __init__(self):
        self._gender_encoder = None
        self._fit = False
        self._load = False

    def fit(self, genders):
        """ Fit the gender label encoder if needed. """
        self._gender_encoder = LabelEncoder()
        self._gender_encoder.fit(list(set(genders)))

        with open(self._encoder_path, 'wb') as f:
            pickle.dump(self._gender_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._fit = True

    def load(self):
        """ Load the pre-fit gender label encoder. """
        with open(self._encoder_path, 'rb') as f:
            self._gender_encoder = pickle.load(f)

        self._load = True

    def encode(self, genders):
        """ Convert gender values to encoded integers. """
        if not self._gender_encoder:
            self.load()

        encoded_genders = self._gender_encoder.transform(genders)

        return encoded_genders


class NameEncoder(object):
    """ Encode the name list into encoded char-to-int 2-D numpy array. """
    _start_char = '^'
    _end_char = '$'
    _char_encoder_file_name = 'char_encoder.pkl'
    _encoder_path = os.path.join(PARENT_DIR, 'models', _char_encoder_file_name)

    def __init__(self, lower=True, pad_size=103, padding='post'):
        self._lower = lower
        self._char_encoder = None
        self._fit = False
        self._load = False
        self._pad_size = pad_size
        self._padding = padding

    def text_clean(self, name):
        """ Clean the input name string. """
        try:
            if self._lower:
                name = name.lower()

            name = re.sub('\\(.*?\\)|\\[.*?\\]|{.*?\\}', '', name)
            name = re.sub('[^\\w \\-"\'.]+', ' ', name)
            name = re.sub('[0-9]', '', name)
            name = ' '.join(name.split())
            return name
        except (TypeError, AttributeError) as e:
            logger.exception(f'text_clean [{name}]: {e}')

    def fit(self, names):
        """ Fit the new encoder if not loaded. """
        clean_names = [self.text_clean(name) for name in names]
        characters = list(set(''.join(clean_names)))
        characters = [self._start_char, self._end_char] + characters
        self._char_encoder = LabelEncoder()
        self._char_encoder.fit(characters)

        with open(self._encoder_path, 'wb') as f:
            pickle.dump(self._char_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._fit = True

    def load(self):
        """ Load the fitted encoder. """
        with open(self._encoder_path, 'rb') as f:
            self._char_encoder = pickle.load(f)
        self._load = True

    def encode(self, names):
        """ Encode all input names. """
        encoded_names = []
        if not self._char_encoder:
            self.load()

        for name in names:
            clean_name = list(self.text_clean(name))
            clean_name = [self._start_char] + clean_name + [self._end_char]
            try:
                encoded_names.append(self._char_encoder.transform(clean_name))
            except Exception as e:
                logger.exception(f'encode [{name}]: {e}')
                clean_name = [char for char in clean_name if char in self._char_encoder.classes_]
                encoded_names.append(self._char_encoder.transform(clean_name))

        encoded_names = pad_sequences(encoded_names, maxlen=self._pad_size, padding=self._padding)

        return encoded_names

    @property
    def char_size(self):
        return len(self._char_encoder.classes_)


class CharBiLSTM(object):
    """ Character-based bi-directional LSTM model. """
    _classifier_file_name = 'gender_model.h5'
    _classifier_backup_name = 'gender_model_backup.h5'
    _classifier_path = os.path.join(PARENT_DIR, 'models', _classifier_file_name)
    _classifier_backup_path = os.path.join(PARENT_DIR, 'models', _classifier_backup_name)
=======
class CharBiLSTM(object):
    """ Character-based bi-directional LSTM model. """
    _classifier_weights_file_name = 'gender_model_weights.h5'
    _classifier_graph_file_name = 'model_graph.json'
    _classifier_weights_next_name = 'gender_model_weights_next.h5'
    _classifier_graph_next_name = 'model_graph_next.json'
    _classifier_weights_path = os.path.join(PARENT_DIR, 'models', _classifier_weights_file_name)
    _classifier_graph_path = os.path.join(PARENT_DIR, 'models', _classifier_graph_file_name)
    _classifier_weights_next_path = os.path.join(PARENT_DIR, 'models', _classifier_weights_next_name)
    _classifier_graph_next_path = os.path.join(PARENT_DIR, 'models', _classifier_graph_next_name)
>>>>>>> 4fe210c4f225ae30b22b0fbeab56621621b768ea

    def __init__(self, lower=True, pad_size=103, padding='post', embedding_size=256, lstm_size1=128, lstm_size2=128,
                 lstm_dropout1=0.2, lstm_dropout2=0.2, output_dim=1, optimizer='adam', loss='binary_crossentropy',
                 metrics=None):
        self._embedding_size = embedding_size
        self._lstm_size1 = lstm_size1
        self._lstm_size2 = lstm_size2
        self._lstm_dropout1 = lstm_dropout1
        self._lstm_dropout2 = lstm_dropout2
        self._output_dim = output_dim
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics if metrics else ['accuracy']
        self._name_encoder = NameEncoder(lower, pad_size, padding)
        self._char_size = None
        self._gender_encoder = GenderEncoder()
        self._model = None

    def _encode_name(self, names, fit=False):
        """ Encode the input names with NameEncoder. """
        if fit:
            self._name_encoder.fit(names)
        encoded_names = self._name_encoder.encode(names)
        self._char_size = self._name_encoder.char_size

        return encoded_names

    def _encode_gender(self, genders, fit=False):
        """ Encode the input genders with GenderEncoder. """
        if fit:
            self._gender_encoder.fit(genders)

        encoded_genders = self._gender_encoder.encode(genders)

        return encoded_genders

<<<<<<< HEAD
    def train(self, names, genders, split_rate=0.2, batch_size=128, patience=5, model_path=_classifier_path,
              save_best_only=True, epochs=100):
=======
    def train(self, names, genders, split_rate=0.2, batch_size=128, patience=5,
              model_weight_path=_classifier_weights_path, model_graph_path=_classifier_graph_path,
              save_best_only=True, save_weights_only=True, epochs=100):
>>>>>>> 4fe210c4f225ae30b22b0fbeab56621621b768ea
        """ Train the LSTM model. """
        names = self._encode_name(names, True)
        genders = self._encode_gender(genders, True)
        X_train, X_valid, y_train, y_valid = train_test_split(names, genders, test_size=split_rate)
        valid_batch_size = min(batch_size, len(X_valid) // 3)
        train_gtr = KerasBatchGenerator(X_train, y_train, batch_size)
        valid_gtr = KerasBatchGenerator(X_valid, y_valid, valid_batch_size)

        earlystop = EarlyStopping(patience=patience)
<<<<<<< HEAD
        checkpoint = ModelCheckpoint(model_path, save_best_only=save_best_only)
=======
        checkpoint = ModelCheckpoint(model_weight_path, save_best_only=save_best_only,
                                     save_weights_only=save_weights_only)
>>>>>>> 4fe210c4f225ae30b22b0fbeab56621621b768ea
        history = History()

        model = Sequential()
        model.add(Embedding(self._name_encoder.char_size, output_dim=self._embedding_size))
        model.add(Bidirectional(LSTM(self._lstm_size1, return_sequences=True)))
        model.add(Dropout(rate=self._lstm_dropout1))
        model.add(Bidirectional(LSTM(self._lstm_size2)))
        model.add(Dropout(rate=self._lstm_dropout2))
        model.add(Dense(self._output_dim, activation='sigmoid'))
        model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)

        model.fit_generator(train_gtr.generate(), len(X_train) // batch_size, epochs=epochs,
                            validation_data=valid_gtr.generate(), validation_steps=len(X_valid) // valid_batch_size,
                            callbacks=[earlystop, checkpoint, history])
        for epoch in np.arange(0, len(model.history.history['loss'])):
            logger.info(f"Epoch={epoch + 1}, "
                        f"{', '.join(f'{key}={value[epoch]}' for key, value in model.history.history.items())}")

<<<<<<< HEAD
    def load(self, model_path=_classifier_path):
        """ Load the existing master model. """
        self._model = load_model(model_path)

    def update(self, names, genders, split_rate=0.2, batch_size=64, patience=1, model_path=_classifier_path,
               save_best_only=True, epochs=2):
        """ This function keep the original model, update the model and save it as default model. """
        names = self._encode_name(names, True)
        genders = self._encode_gender(genders, True)
=======
        # Save the model structure.
        with open(model_graph_path, 'w') as f:
            f.write(model.to_json())

        # Load the trained model.
        self._model = model

    def load(self, model_weights_path=_classifier_weights_path, model_graph_path=_classifier_graph_path):
        """ Load the existing master model. """
        K.clear_session()
        with open(model_graph_path, 'r') as f:
            model_graph = f.read()
        self._model = model_from_json(model_graph)
        self._model.load_weights(model_weights_path)

    def update(self, names, genders, split_rate=0.2, batch_size=64, patience=1,
               model_weights_next_path=_classifier_weights_next_path,
               model_graph_next_path=_classifier_graph_next_path,
               save_best_only=True, save_weights_only=True, epochs=2):
        """ This function keep the original model, update the model and save it as default model. """
        names = self._encode_name(names)
        genders = self._encode_gender(genders)
>>>>>>> 4fe210c4f225ae30b22b0fbeab56621621b768ea
        X_train, X_valid, y_train, y_valid = train_test_split(names, genders, test_size=split_rate)
        valid_batch_size = min(batch_size, len(X_valid) // 3)
        train_gtr = KerasBatchGenerator(X_train, y_train, batch_size)
        valid_gtr = KerasBatchGenerator(X_valid, y_valid, valid_batch_size)

        earlystop = EarlyStopping(patience=patience)
<<<<<<< HEAD
        checkpoint = ModelCheckpoint(model_path, save_best_only=save_best_only)
=======
        checkpoint = ModelCheckpoint(model_weights_next_path, save_best_only=save_best_only,
                                     save_weights_only=save_weights_only)
>>>>>>> 4fe210c4f225ae30b22b0fbeab56621621b768ea
        history = History()

        if not self._model:
            self.load()
<<<<<<< HEAD
        self._model.save(self._classifier_backup_path)
=======

>>>>>>> 4fe210c4f225ae30b22b0fbeab56621621b768ea
        self._model.fit_generator(train_gtr.generate(), len(X_train) // batch_size, epochs=epochs,
                                  validation_data=valid_gtr.generate(), validation_steps=len(X_valid) // valid_batch_size,
                                  callbacks=[earlystop, checkpoint, history])
        for epoch in np.arange(0, len(self._model.history.history['loss'])):
            logger.info(f"Epoch={epoch + 1}, "
                        f"{', '.join(f'{key}={value[epoch]}' for key, value in self._model.history.history.items())}")

<<<<<<< HEAD
=======
        # Save the model structure.
        with open(model_graph_next_path, 'w') as f:
            f.write(self._model.to_json())

    def overwrite(self):
        """This function copy the next model version to overwrite the current version."""
        move_file = Popen(f'cp {self._classifier_weights_next_path} {self._classifier_weights_path}; '
                          f'cp {self._classifier_graph_next_path} {self._classifier_graph_path}',
                          shell=True, stdout=PIPE, executable='/bin/bash')
        move_file.communicate()

>>>>>>> 4fe210c4f225ae30b22b0fbeab56621621b768ea
    def predict(self, names, return_prob=False, ptv_cutoff=gender_cutoff[1], ntv_cutoff=gender_cutoff[0]):
        """ This function predicts the gender with given names. """
        if not self._model:
            self.load()
        names = self._encode_name(names)
        y_pred_prob = self._model.predict(names)
        y_pred_prob = y_pred_prob.flatten()
        y_pred = np.where(y_pred_prob >= ptv_cutoff, gender_class[1],
                          np.where(y_pred_prob >= ntv_cutoff, gender_class['unk'], gender_class[0]))

        if return_prob:
            return [{'gender': pred, gender_class[1]: prob, gender_class[0]: 1 - prob}
                    for pred, prob in zip(y_pred, y_pred_prob)]
        else:
            return y_pred.tolist()
