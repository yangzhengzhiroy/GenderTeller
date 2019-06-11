`GenderTeller` is meant to replace the existing implemented package `chicksexer` due to obsolete dependencies.

## Installation

```bash
tar -xvf GenderTeller*.tar.gz
cd GenderTeller*/
python setup.py install
```

## API usage.
```bash
from genderteller import predict_gender, predict_genders

>>> predict_genders(['Angela Dorothea Merkel', 'Donald Trump', 'Lee Hsien Loong'])
['Female', 'Male', 'Male']

>>> predict_genders(['Angela Dorothea Merkel', 'Donald Trump', 'Lee Hsien Loong'], return_prob=True)
[{'gender': 'Female', 'Male': 0.00089112035, 'Female': 0.99910887965234},
 {'gender': 'Male', 'Male': 0.99972254, 'Female': 0.00027745962142944336},
 {'gender': 'Male', 'Male': 0.9145421, 'Female': 0.08545792102813721}]

>>> predict_gender('Angela Dorothea Merkel')
'Female'

```