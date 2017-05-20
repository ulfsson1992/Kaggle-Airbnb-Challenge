import pandas as pd


# Prints out all different values in a column and the number of occurences
def analyze_column(df, col):
    g = df.groupby(col, as_index=False).count()
    return g

# Method for normalizing the values in a list of columns
def normalize_cols(df, cols):
    df[cols] = df[cols].div(df[cols].sum(axis=1), axis=0)
    df[cols] = df[cols].fillna(0)

session_url = "sessions.csv"

session_df = pd.read_csv(session_url, index_col=False)

# Fill missing data
session_df[['action', 'action_type', 'action_detail']] = session_df[['action', 'action_type', 'action_detail']].fillna('NaN')
session_df['secs_elapsed'] = session_df['secs_elapsed'].fillna(0)

### Format action data
action_df = session_df[['action', 'action_type', 'action_detail', 'user_id']]

# count the amount of performed actions for every user
g = action_df.groupby(['action', 'action_type', 'action_detail'], as_index=False).count()
g = g[['action', 'action_type', 'action_detail', 'user_id']]
g.columns = ['action', 'action_type', 'action_detail', 'action_count']

action_df = pd.merge(action_df, g, how='left', on=['action', 'action_type', 'action_detail'])

# Group together all action fields
action_df['action_gr'] = action_df['action'] + action_df['action_type'] + action_df['action_detail']

# Remove redundant fields
action_df.drop(['action', 'action_type', 'action_detail'], axis=1, inplace=True)

# Group together all actions used less than .1% of the time by renaming it to binned
action_df.loc[(action_df['action_count'] < 20000), 'action_gr'] = 'binned'

# Drop uneccesary rows for dealing with cleaning of device data
action_df = action_df[['user_id', 'action_gr', 'action_count']].drop_duplicates()

action_list = action_df['action_gr'].drop_duplicates().values

# Fix action names
action_df['action_gr'] = action_df['action_gr'].replace(" ","_").replace("-","").replace("/","").replace("(","").replace(")","")

print('Starting to split the action feature')

for action in action_list:
    action_df[action] = 0
    action_df.loc[(action_df['action_gr'] == action), action] = action_df['action_count']

action_df.drop(['action_gr', 'action_count'], axis=1, inplace=True)

action_df = action_df.groupby(['user_id'], as_index=False).sum()

# Normalize the list of actions for every user
normalize_cols(action_df, action_list)

### Format Device data
device_df = session_df[['user_id', 'device_type', 'secs_elapsed']]

# Split the device type feature into several of its component features containing the duration
# that each device was used
device_list = device_df['device_type'].drop_duplicates().values

for device in device_list:
    device_df[device] = 0
    device_df.loc[(device_df['device_type'] == device), device] = device_df['secs_elapsed']

# Drop the source features
device_df.drop(['device_type', 'secs_elapsed'], axis=1, inplace=True)

# Group each user and summarise the device duration
device_df = device_df.groupby(['user_id'], as_index=False).sum()

# normalize the device list columns for every user
normalize_cols(device_df, device_list)

# Merge the device dataframe with the action dataframe
session_df = pd.merge(action_df, device_df, how='outer', on='user_id')

session_df.to_csv('Cleaned Data-Sets/Cleaned_sessions.csv', index=False)