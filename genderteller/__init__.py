# -*- coding: UTF-8 -*-
"""
GenderTeller package
"""
import os


__author__ = 'yang zhengzhi'


PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

gender_class = {1: 'Male', 0: 'Female', 'unk': 'Undisclosed'}

gender_cutoff = {1: 0.7, 0: 0.25}

from .api import predict_genders, predict_gender