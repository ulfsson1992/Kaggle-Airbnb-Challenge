import pandas as pd
import numpy as np
import pickle

from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder

## Prepare for prediction
users_url = 'Cleaned Data-Sets/all_users.csv'
sessions_url = 'Cleaned Data-Sets/Cleaned_sessions.csv'

users_df = pd.read_csv(users_url, index_col=False, low_memory=False)
sessions_df = pd.read_csv(sessions_url, index_col=False, low_memory=False)

sessions_df['id'] = sessions_df['user_id']
sessions_df.drop('user_id', axis=1, inplace=True)

all_df = pd.merge(sessions_df, users_df, how='inner', on='id')
test_users = pd.read_csv('Cleaned Data-Sets/test_users.csv')[['id']]

model = pickle.load(open('model.p','rb'))
le = pickle.load(open('labelencoder.p', 'rb'))

# Inner join it with the all data frame should only keep test users
test_df = pd.merge(test_users, all_df, on='id')
test_df.set_index('id', inplace=True)
test_df.drop('country_destination', axis=1, inplace=True)
id_list = test_df.index.values

y_pred = model.predict_proba(test_df)

## Store prediction according to Kaggle format
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_list)):
    idx = id_list[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
print("Outputting final results...")
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('submission.csv', index=False)