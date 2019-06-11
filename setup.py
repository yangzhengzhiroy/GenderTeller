import os
from setuptools import setup, find_packages


__version__ = '0.1.0'
__author__ = 'yang zhengzhi'
__email__ = 'yangzhengzhi.roy@gmail.com'

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(ROOT_DIR, 'requirements.txt'), 'r') as f:
    requirements = f.read().split('\n')

setup(
    name='GenderTeller',
    version=__version__,
    description='Predict gender based on the name',
    author=__author__,
    author_email=__email__,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Topic :: Data Science :: Natural Language Processing',
        'Topic :: Machine Learning :: Deep Neural Network'
    ],
    packages=find_packages(),
    install_requires=requirements,
<<<<<<< HEAD
    include_package_data=True,
    package_data={
        '': ['models/*']
    },
    data_files=[('models', ['char_encoder.pkl', 'gender_encoder.pkl', 'gender_model.h5'])]
=======
    include_package_data=True
>>>>>>> 4fe210c4f225ae30b22b0fbeab56621621b768ea
)
