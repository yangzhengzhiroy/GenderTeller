# -*- coding: UTF-8 -*-
"""
This module prepares the input names and gender label.
"""
import os
import re
import pickle
import logging
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from genderteller import PARENT_DIR


logger = logging.getLogger(__name__)


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
