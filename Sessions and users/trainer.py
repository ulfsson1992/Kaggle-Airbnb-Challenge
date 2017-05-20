import pandas as pd
import xgboost as xgb
import pickle

from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot

users_url = 'users.csv'
sessions_url = 'cleaned_sessions.csv'

users_df = pd.read_csv(users_url, index_col=False, low_memory=False)
sessions_df = pd.read_csv(sessions_url, index_col=False, low_memory=False)

sessions_df['id'] = sessions_df['user_id']
sessions_df.drop('user_id', axis=1, inplace=True)

all_df = pd.merge(sessions_df, users_df, how='left', on='id')
temp = all_df['country_destination']

# before filling up all empty slots with 0's we could add their first device as a 1
all_df = all_df.fillna(0)

all_df['country_destination'] = temp

## Prepare user set for training
#basically just remove all entries with no specified destination country

train_df = all_df.dropna()
train_df.set_index('id', inplace=True)

id_train = train_df.index.values
labels = train_df['country_destination']
le = LabelEncoder()
y = le.fit_transform(labels)
X = train_df.drop('country_destination', axis=1)

print(len(X.columns))
print(len(sessions_df.columns))
print(sessions_df.columns)

# ## Train the classifier
# model = xgb.XGBClassifier(objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0, learning_rate=0.1, n_estimators=150)
# # param_grid = {'max_depth': [3, 4, 5], 'learning_rate': [0.1, 0.3], 'n_estimators': [50, 100]}
# # model = model_selection.GridSearchCV(estimator=XGB_model, param_grid=param_grid, scoring='accuracy', verbose=10, iid=True, refit=True, cv=3)
#
# model.fit(X, y)
# # print("Best score: %0.3f" % model.best_score_)
# # print("Best parameters set:")
# # best_parameters = model.best_estimator_.get_params()
# # for param_name in sorted(param_grid.keys()):
# #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
#
# xgb.plot_importance(model)
#
# ## Store the model and the label encoder in a pickle
# pickle.dump(model, open('model.p', 'wb'))
# pickle.dump(le, open('labelencoder.p', 'wb'))
