import os
import re
import sys
import json
import pandas as pd
from glob import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


name_dict = {'male': [], 'female': []}
files = glob('/home/zhengzhi/Downloads/pdldata/*')
for file in files:
    print(os.path.basename(file))
    with open(file, 'r') as f:
        documents = f.read().split('\n')
    documents = list(filter(None, documents))
    for document in documents:
        document = json.loads(document)
        gender = document.get('GenderTeller')
        name = document.get('names')
        if not name:
            continue
        name = name[0]
        first_name = name.get('first_name')
        last_name = name.get('last_name')
        full_name = name.get('name')
        if gender not in name_dict:
            name_dict[gender] = []
        name_dict[gender].append({'first_name': first_name, 'last_name': last_name, 'full_name': full_name})
with open('data/pdl_name_gender.json', 'w') as f:
    json.dump(name_dict, f)

full_data_df = pd.DataFrame()

# persondata_en.tql
with open('data/persondata_en.tql', 'r') as f:
    data = f.read()
data = data.split('\n')
data = data[1:-2]
data = [[re.search('<.*?>', entry).group(), entry] for entry in data]
gender = [item for item in data if 'http://xmlns.com/foaf/0.1/gender' in item[1].lower()]
gender = [[item[0], re.search('".*?"', item[1]).group()] for item in gender]
gender = [[item[0], re.sub('["+]', '', item[1])] for item in gender]
full_name = [item for item in data if 'http://xmlns.com/foaf/0.1/name' in item[1].lower()]
full_name = [[item[0], re.search('".*?"', item[1]).group()] for item in full_name]
full_name = [[item[0], re.sub('["+]', '', item[1])] for item in full_name]
first_name = [item for item in data if 'http://xmlns.com/foaf/0.1/givenname' in item[1].lower()]
first_name = [[item[0], re.search('".*?"', item[1]).group()] for item in first_name]
first_name = [[item[0], re.sub('["+]', '', item[1])] for item in first_name]
last_name = [item for item in data if 'http://xmlns.com/foaf/0.1/surname' in item[1].lower()]
last_name = [[item[0], re.search('".*?"', item[1]).group()] for item in last_name]
last_name = [[item[0], re.sub('["+]', '', item[1])] for item in last_name]
gender = pd.DataFrame(gender, columns=['id', 'GenderTeller'])
full_name = pd.DataFrame(full_name, columns=['id', 'full_name'])
first_name = pd.DataFrame(first_name, columns=['id', 'first_name'])
last_name = pd.DataFrame(last_name, columns=['id', 'last_name'])
data_df = gender.merge(full_name, how='left', on='id')
data_df = data_df.merge(first_name, how='left', on='id')
data_df = data_df.merge(last_name, how='left', on='id')
data_df['id'] = 'dbpedia'
full_data_df = full_data_df.append(data_df, ignore_index=True, sort=False)

# US baby names.
files = glob('data/us_name_states/*.TXT')
data_df = pd.DataFrame()
for file in files:
    with open(file, 'r') as f:
        document = f.read().split('\n')
    document = document[:-1]
    document = [item.split(',') for item in document]
    document = pd.DataFrame(document, columns=['state', 'GenderTeller', 'year', 'first_name', 'count'])
    document = document.drop_duplicates(subset=['GenderTeller', 'first_name'])
    data_df = data_df.append(document, ignore_index=True, sort=False)
data_df = data_df.reset_index(drop=True)
data_df = data_df[pd.notnull(data_df['first_name'])]
data_df['full_name'] = data_df['first_name']
data_df['GenderTeller'] = data_df['GenderTeller'].map({'F': 'female', 'M': 'male'})
data_df = data_df[['GenderTeller', 'first_name', 'full_name']]
data_df['id'] = 'us_baby'

full_data_df = full_data_df.append(data_df, ignore_index=True, sort=False)

# mbejda.
# Black F.
data = pd.read_csv('data/Black-Female-Names.csv')
data = data.rename(columns={' first name': 'first_name', 'last name': 'last_name'})
data['full_name'] = data['first_name'].str.strip() + ' ' + data['last_name'].str.strip()
data['id'] = 'mbejda'
data['GenderTeller'] = 'female'
data = data[['id', 'GenderTeller', 'full_name', 'first_name', 'last_name']]
full_data_df = full_data_df.append(data, ignore_index=True, sort=False)
# Black M.
data = pd.read_csv('data/Black-Male-Names.csv')
data = data.rename(columns={'first name': 'first_name', 'last name': 'last_name'})
data['full_name'] = data['first_name'].str.strip() + ' ' + data['last_name'].str.strip()
data['id'] = 'mbejda'
data['GenderTeller'] = 'male'
data = data[['id', 'GenderTeller', 'full_name', 'first_name', 'last_name']]
full_data_df = full_data_df.append(data, ignore_index=True, sort=False)
# Hispanic F.
data = pd.read_csv('data/Hispanic-Female-Names.csv')
data = data.rename(columns={' first name': 'first_name', 'last name': 'last_name'})
data['full_name'] = data['first_name'].str.strip() + ' ' + data['last_name'].str.strip()
data['id'] = 'mbejda'
data['GenderTeller'] = 'female'
data = data[['id', 'GenderTeller', 'full_name', 'first_name', 'last_name']]
full_data_df = full_data_df.append(data, ignore_index=True, sort=False)
# Hispanic M.
data = pd.read_csv('data/Hispanic-Male-Names.csv')
data = data.rename(columns={'first name': 'first_name', 'last name': 'last_name'})
data['full_name'] = data['first_name'].str.strip() + ' ' + data['last_name'].str.strip()
data['id'] = 'mbejda'
data['GenderTeller'] = 'male'
data = data[['id', 'GenderTeller', 'full_name', 'first_name', 'last_name']]
full_data_df = full_data_df.append(data, ignore_index=True, sort=False)
# Indian F.
data = pd.read_csv('data/Indian-Female-Names.csv')
data = data.rename(columns={'name': 'full_name'})
data['id'] = 'mbejda'
data['GenderTeller'] = 'female'
data = data[['id', 'GenderTeller', 'full_name']]
full_data_df = full_data_df.append(data, ignore_index=True, sort=False)
# Indian M.
data = pd.read_csv('data/Indian-Male-Names.csv')
data = data.rename(columns={'name': 'full_name'})
data['id'] = 'mbejda'
data['GenderTeller'] = 'male'
data = data[['id', 'GenderTeller', 'full_name']]
full_data_df = full_data_df.append(data, ignore_index=True, sort=False)
# White F.
data = pd.read_csv('data/White-Female-Names.csv')
data = data.rename(columns={' first name': 'first_name', 'last name': 'last_name'})
data['full_name'] = data['first_name'].str.strip() + ' ' + data['last_name'].str.strip()
data['id'] = 'mbejda'
data['GenderTeller'] = 'female'
data = data[['id', 'GenderTeller', 'full_name', 'first_name', 'last_name']]
full_data_df = full_data_df.append(data, ignore_index=True, sort=False)
# White M.
data = pd.read_csv('data/White-Male-Names.csv')
data = data.rename(columns={' first name': 'first_name', 'last name': 'last_name'})
data['full_name'] = data['first_name'].str.strip() + ' ' + data['last_name'].str.strip()
data['id'] = 'mbejda'
data['GenderTeller'] = 'male'
data = data[['id', 'GenderTeller', 'full_name', 'first_name', 'last_name']]
full_data_df = full_data_df.append(data, ignore_index=True, sort=False)

# Two datasets.
with open('data/female.txt', 'r') as f:
    female = f.read().split('\n')
female = female[:-1]
female = [['misc', 'female', item] for item in female]
female = pd.DataFrame(female, columns=['id', 'GenderTeller', 'full_name'])
with open('data/male.txt', 'r') as f:
    male = f.read().split('\n')
male = male[:-1]
male = [['misc', 'male', item] for item in male]
male = pd.DataFrame(male, columns=['id', 'GenderTeller', 'full_name'])
full_data_df = full_data_df.append(female, ignore_index=True, sort=False)
full_data_df = full_data_df.append(male, ignore_index=True, sort=False)

# top names.
with open('data/female_name_loc.json', 'r') as f:
    data = json.load(f)
names = [item for subset in data.values() for item in subset]
names = [['topname', 'female', item] for item in names]
names = pd.DataFrame(names, columns=['id', 'GenderTeller', 'full_name'])
full_data_df = full_data_df.append(names, ignore_index=True, sort=False)
with open('data/male_name_loc.json', 'r') as f:
    data = json.load(f)
names = [item for subset in data.values() for item in subset]
names = [['topname', 'male', item] for item in names]
names = pd.DataFrame(names, columns=['id', 'GenderTeller', 'full_name'])
full_data_df = full_data_df.append(names, ignore_index=True, sort=False)

# pdldata.
with open('data/pdl_name_gender.json', 'r') as f:
    data = json.load(f)
data = {key: value for key, value in data.items() if key != 'null'}
female = [['pdldata', 'female', item.get('full_name'), item.get('first_name'), item.get('last_name')] for item in data['female']]
male = [['pdldata', 'male', item.get('full_name'), item.get('first_name'), item.get('last_name')] for item in data['male']]
female = pd.DataFrame(female, columns=['id', 'GenderTeller', 'full_name', 'first_name', 'last_name'])
male = pd.DataFrame(male, columns=['id', 'GenderTeller', 'full_name', 'first_name', 'last_name'])
full_data_df = full_data_df.append(female, ignore_index=True, sort=False)
full_data_df = full_data_df.append(male, ignore_index=True, sort=False)

# further clean.
full_data_df = full_data_df[full_data_df['GenderTeller'].isin(('male', 'female'))]
full_data_df['GenderTeller'] = full_data_df['GenderTeller'].str.title()
full_data_df = full_data_df[['full_name', 'GenderTeller']]
full_data_df = full_data_df.dropna()
full_data_df['full_name'] = full_data_df['full_name'].apply(lambda x: re.sub('\\(.*?\\)|\\[.*?\\]|{.*?\\}', '', x))
full_data_df['full_name'] = full_data_df['full_name'].apply(lambda x: re.sub('[^\\w \\-"\'.]+', ' ', x))
full_data_df['full_name'] = full_data_df['full_name'].str.strip().str.lower()
full_data_df['full_name'] = full_data_df['full_name'].apply(lambda x: ' '.join(x.split()))
full_data_df['full_name'] = full_data_df['full_name'].apply(lambda x: re.sub('[0-9]', '', x))
full_data_df = full_data_df.dropna()
