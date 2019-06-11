# -*- coding: UTF-8 -*-
"""
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


setup_logging(log_config)
logger = logging.getLogger(__name__)


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

    def train(self, names, genders, split_rate=0.2, batch_size=128, patience=5,
              model_weight_path=_classifier_weights_path, model_graph_path=_classifier_graph_path,
              save_best_only=True, save_weights_only=True, epochs=100):
        """ Train the LSTM model. """
        names = self._encode_name(names, True)
        genders = self._encode_gender(genders, True)
        X_train, X_valid, y_train, y_valid = train_test_split(names, genders, test_size=split_rate)
        valid_batch_size = min(batch_size, len(X_valid) // 3)
        train_gtr = KerasBatchGenerator(X_train, y_train, batch_size)
        valid_gtr = KerasBatchGenerator(X_valid, y_valid, valid_batch_size)

        earlystop = EarlyStopping(patience=patience)
        checkpoint = ModelCheckpoint(model_weight_path, save_best_only=save_best_only,
                                     save_weights_only=save_weights_only)
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
        X_train, X_valid, y_train, y_valid = train_test_split(names, genders, test_size=split_rate)
        valid_batch_size = min(batch_size, len(X_valid) // 3)
        train_gtr = KerasBatchGenerator(X_train, y_train, batch_size)
        valid_gtr = KerasBatchGenerator(X_valid, y_valid, valid_batch_size)

        earlystop = EarlyStopping(patience=patience)
        checkpoint = ModelCheckpoint(model_weights_next_path, save_best_only=save_best_only,
                                     save_weights_only=save_weights_only)
        history = History()

        if not self._model:
            self.load()

        self._model.fit_generator(train_gtr.generate(), len(X_train) // batch_size, epochs=epochs,
                                  validation_data=valid_gtr.generate(), validation_steps=len(X_valid) // valid_batch_size,
                                  callbacks=[earlystop, checkpoint, history])
        for epoch in np.arange(0, len(self._model.history.history['loss'])):
            logger.info(f"Epoch={epoch + 1}, "
                        f"{', '.join(f'{key}={value[epoch]}' for key, value in self._model.history.history.items())}")

        # Save the model structure.
        with open(model_graph_next_path, 'w') as f:
            f.write(self._model.to_json())

    def overwrite(self):
        """This function copy the next model version to overwrite the current version."""
        move_file = Popen(f'cp {self._classifier_weights_next_path} {self._classifier_weights_path}; '
                          f'cp {self._classifier_graph_next_path} {self._classifier_graph_path}',
                          shell=True, stdout=PIPE, executable='/bin/bash')
        move_file.communicate()

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
