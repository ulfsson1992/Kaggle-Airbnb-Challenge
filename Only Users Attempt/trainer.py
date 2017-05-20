import pandas as pd
import xgboost as xgb
import pickle

from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder

users_url = 'users.csv'

users_df = pd.read_csv(users_url, index_col=False, low_memory=False)

## Prepare user set for training
#basically just remove all entries with no specified destination country

train_df = users_df.dropna()
train_df.set_index('id', inplace=True)

id_train = train_df.index.values
labels = train_df['country_destination']
le = LabelEncoder()
y = le.fit_transform(labels)
X = train_df.drop('country_destination', axis=1)

print(len(X.columns))

# ## Train the classifier
# XGB_model = xgb.XGBClassifier(objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
# param_grid = {'max_depth': [3, 4, 5], 'learning_rate': [0.1, 0.3], 'n_estimators': [25, 50]}
# model = model_selection.GridSearchCV(estimator=XGB_model, param_grid=param_grid, scoring='accuracy', verbose=10, n_jobs=1, iid=True, refit=True, cv=3)
#
# model.fit(X, y)
# print("Best score: %0.3f" % model.best_score_)
# print("Best parameters set:")
# best_parameters = model.best_estimator_.get_params()
# for param_name in sorted(param_grid.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
#
# ## Store the model and the label encoder in a pickle
# pickle.dump(model, open('model.p', 'wb'))
# pickle.dump(le, open('labelencoder.p', 'wb'))
