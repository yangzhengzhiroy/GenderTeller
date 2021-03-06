import logging
from .utils import log_config, setup_logging
from .classifier import GenderModel
from genderteller import gender_cutoff, gender_class


setup_logging(log_config)
logger = logging.getLogger(__name__)
cur_model = GenderModel()


def predict_genders(names, return_prob=False, ptv_cutoff=gender_cutoff[1], ntv_cutoff=gender_cutoff[0]):
    """ The function predicts genders and probabilities based on names. """
    try:
        global _model
        _model = GenderModel.load()
        if names:
            return cur_model.predict(_model, names, return_prob, ptv_cutoff, ntv_cutoff)
    except Exception as e:
        logger.exception(f'predict_genders: {e}')


def predict_gender(name, return_prob=False, ptv_cutoff=gender_cutoff[1], ntv_cutoff=gender_cutoff[0]):
    try:
        output = predict_genders([name], return_prob, ptv_cutoff, ntv_cutoff)
        if output:
            return output[0]
        else:
            return gender_class['unk']
    except Exception as e:
        logger.exception(f'predict_gender: {e}')
