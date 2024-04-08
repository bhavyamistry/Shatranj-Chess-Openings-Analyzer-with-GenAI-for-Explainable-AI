#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import catboost as cb
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import RBFSampler
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# # Dataset

# In[5]:


chess_data_df = pd.read_csv("../input/chess-games/chess_games.csv")
print(chess_data_df.head())


# 

# # Correlation Matrix

# In[58]:


# calculate correlation matrix
corr_matrix = chess_data_df.corr()

# plot correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='rocket')
plt.title('Correlation Matrix')
plt.show()


# In[59]:


# According to the correlation Matrix columns 'WhiteRatingDiff' and  'BlackRatingDiff' 
# have least correlation and the values just the meant the what difference in rank affects the game
# If wins difference is positive or else difference is negative


# In[60]:


plt.figure(figsize=(8, 7))
custom_palette = ["#8B0000", "#FF8C00", "#FFD700", "#228B22", "#00BFFF", "#1E90FF"]
sns.countplot(y='Event', data=chess_data_df, palette=custom_palette, order=chess_data_df['Event'].value_counts().index)

# set x-axis label and scale
plt.xlabel('Count')
plt.ticklabel_format(axis='x', style='plain', useOffset=False)

# set x-axis limits and tick values
plt.xlim(0, 3000000)
plt.xticks([0, 500000, 1000000, 1500000, 2000000, 2500000,3000000], 
           ['0', '0.5M', '1M', '1.5M', '2M', '2.5M','3M'])

# display plot
plt.show()


# In[61]:


import pandas as pd
import matplotlib.pyplot as plt

# create a new dataframe with count of each event
event_counts = chess_data_df['Event'].value_counts().reset_index()
event_counts.columns = ['Event', 'Count']

# create a pie chart
fig, ax = plt.subplots(figsize=(10,10))
ax.pie(event_counts['Count'], labels=event_counts['Event'], autopct='%1.1f%%', startangle=90, counterclock=False)
ax.set_title('Event Distribution')
plt.show()


# In[62]:


plt.figure(figsize=(10, 5))
custom_palette = ["#8B0000", "#FF8C00", "#FFD700", "#228B22", "#00BFFF", "#1E90FF"]
ax = sns.countplot(x='Termination', data=chess_data_df, palette=custom_palette)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 3, str(round(height/1000000,2))+'M', ha='center')
# set y-axis label and scale
# set y-axis label and scale
plt.ylabel('Count')
plt.ticklabel_format(axis='y', style='plain', useOffset=False)

# set y-axis limits and tick values
plt.ylim(0, 6000000)
plt.yticks([0, 1000000, 2000000, 3000000, 4000000, 5000000], 
           ['0', '1M', '2M', '3M', '4M', '5M'])
plt.show()


# In[63]:


plt.figure(figsize=(10, 5))
custom_palette = ["#8B0000", "#FF8C00", "#FFD700", "#228B22", "#00BFFF", "#1E90FF"]
ax = sns.countplot(x='Result', data=chess_data_df, palette=custom_palette)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 3, str(round(height/1000000,2))+'M', ha='center')
# set y-axis label and scale
# set y-axis label and scale
plt.ylabel('Count')
plt.ticklabel_format(axis='y', style='plain', useOffset=False)

# set y-axis limits and tick values
plt.ylim(0, 6000000)
plt.yticks([0, 1000000, 2000000, 3000000, 4000000, 5000000], 
           ['0', '1M', '2M', '3M', '4M', '5M'])
plt.show()


# In[64]:


# Set up the figure size and style
sns.set(rc={'figure.figsize':(10,5)})
sns.set_style("whitegrid")

# Create a list of the two columns to plot
data = [chess_data_df['WhiteElo'], chess_data_df['BlackElo']]

# Create the box plot
bp = sns.boxplot(data=data, palette='rocket', showfliers=True, width=0.5)
bp.set(xlabel='User Ratings', xticklabels=['WhiteElo', 'BlackElo'])

# Show the plot
plt.show()


# In[65]:


sns.histplot(chess_data_df['WhiteElo'],palette='rocket')


# In[66]:


sns.histplot(chess_data_df['BlackElo'])


# 

# # Preprocessing

# In[88]:


new_chess_data_df = chess_data_df


# In[89]:


# Row-wise missing value analysis
missing_values_row = new_chess_data_df.isnull().sum(axis=1)
print("Number of rows with missing values:", len(missing_values_row[missing_values_row > 0]))
print("Percentage of rows with missing values:", round(len(missing_values_row[missing_values_row > 0]) / len(chess_data_df) * 100, 2), "%")

# Column-wise missing value analysis
missing_values_column = new_chess_data_df.isnull().sum()
print("Number of columns with missing values:", len(missing_values_column[missing_values_column > 0]))
print("Percentage of columns with missing values:", round(len(missing_values_column[missing_values_column > 0]) / len(chess_data_df.columns) * 100, 2), "%")

# Overall missing value analysis
total_missing_values = new_chess_data_df.isnull().sum().sum()
print("Total number of missing values:", total_missing_values)
print("Percentage of missing values:", round(total_missing_values / (len(chess_data_df) * len(chess_data_df.columns)) * 100, 2), "%")


# In[90]:


# Drop all rows with NaNs
new_chess_data_df.dropna(inplace=True)
# Print the shape of the cleaned dataframe
print("Shape of dataframe after dropping NaNs:", new_chess_data_df.shape)


# In[91]:


# Drop the columns
new_chess_data_df.drop(['White', 'Black', 'UTCDate', 'UTCTime', 'WhiteRatingDiff', 'BlackRatingDiff', 'TimeControl'], axis=1, inplace=True)
# Print the updated dataframe
# print(chess_data_df.head())
print(new_chess_data_df.shape)
# filter the rows based on the condition
new_chess_data_df = new_chess_data_df[(new_chess_data_df['AN'].str.len() >= 40) & (~new_chess_data_df['AN'].str.contains('{'))]
# reset the index after dropping the rows
new_chess_data_df.reset_index(drop=True, inplace=True)
print(new_chess_data_df.shape)


# In[92]:


print(new_chess_data_df.columns)


# In[93]:


an_val = new_chess_data_df['AN']
# initialize dataframe with empty values
#print(an_val[0])
# loop through each game and extract the first 3 moves
dflist = []
for i, game in enumerate(an_val):
    itr = 0
    temp = []
    flag = 1
    tstr = ''
    if(len(game)<40 or '{' in game):
        continue
    #print(len(game), game)
    while(flag and itr<len(game)):
        if game[itr] == " ":
            temp.append(tstr)
            tstr = ''
        else:
            tstr += game[itr]
        itr+=1
        if(itr+1<len(game)):
            if(game[itr]=='4' and game[itr+1]=='.'):
                flag = 0
    #print(temp)
    temp1 = []
    for i in range(0,len(temp)):
        if i!=0 and i!=3 and i!=6:
            temp1.append(temp[i])
    dflist.append(temp1)
df = pd.DataFrame(dflist, columns=['w1', 'b1', 'w2', 'b2', 'w3', 'b3'])
print(df)


# In[94]:


# merge the two dataframes on their indices
merged_df = pd.merge(new_chess_data_df, df, left_index=True, right_index=True)
print(merged_df)


# In[95]:


new_df = merged_df
new_df.drop(['Event', 'AN', 'Opening', 'Termination', 'ECO'], axis=1, inplace=True)
new_df


# In[96]:


# create a mapping dictionary to replace the values
mapping_dict = {'1-0': 1, '0-1': 0, '1/2-1/2': 2}

# replace the values in the 'Result' column using the mapping dictionary
new_df['Result'] = new_df['Result'].replace(mapping_dict)
new_df


# In[97]:


new_df['Result'].unique()


# In[98]:


new_df = new_df.loc[new_df['Result'] != 2]
new_df = new_df.loc[new_df['Result'] != '*']
print(new_df)
new_df['Result'].unique()


# In[99]:


from sklearn.model_selection import train_test_split
import pandas as pd

# Create a new dataframe with only the features and target variable
X = new_df.drop("Result", axis=1)
y = new_df["Result"]

# Use train_test_split to get a stratified sample of 50000 rows
X, _, y, _ = train_test_split(X, y, train_size=50000, stratify=y)

# Print the value counts of the target variable in the stratified sample
print(y.value_counts())


# In[100]:


print(X)
print(y)


# In[101]:


new_df = pd.concat([X, y], axis=1)
new_df = new_df.reset_index(drop = True)
new_df


# In[102]:


from sklearn.preprocessing import OneHotEncoder
# create an instance of the OneHotEncoder
onehot_encoder = OneHotEncoder()

# fit the encoder on the columns 'w1', 'b1', 'w2', 'b2', 'w3', 'b3'
onehot_encoder.fit(new_df[['w1', 'b1', 'w2', 'b2', 'w3', 'b3']].values)

# transform the columns using one-hot encoding
onehot_encoded = onehot_encoder.transform(new_df[['w1', 'b1', 'w2', 'b2', 'w3', 'b3']].values)

# create a new dataframe with the one-hot encoded columns
onehot_df = pd.DataFrame(onehot_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(['w1', 'b1', 'w2', 'b2', 'w3', 'b3']))

# concatenate the new dataframe with the original dataframe
new_df = pd.concat([new_df, onehot_df], axis=1)

# drop the original columns 'w1', 'b1', 'w2', 'b2', 'w3', 'b3'
new_df = new_df.drop(['w1', 'b1', 'w2', 'b2', 'w3', 'b3'], axis=1)

# print the modified dataframe
print(new_df)


# In[103]:


new_df = new_df.dropna()
print(new_df)


# In[104]:


X = new_df
Y = new_df['Result']
X.drop(['Result'], axis = 1, inplace = True)
print(X)
print(Y)


# In[84]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, stratify = Y)


# In[85]:


from sklearn.naive_bayes import GaussianNB
GNBclf = GaussianNB()
model = GNBclf.fit(X_train, Y_train)
preds = GNBclf.predict(X_test)
print(preds)


# In[109]:


from sklearn.metrics import classification_report
print(classification_report(preds, Y))


# In[80]:


ytest = Y_test.values
ytest = ytest.astype(int)
ytest


# In[82]:


preds = preds.astype(int)
preds


# In[83]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(ytest, preds)
print("Accuracy:", accuracy)


# In[ ]:


# Filter the rows with Termination value Abandoned or Rule Infraction and drop them
new_chess_data_df = new_chess_data_df[~new_chess_data_df['Termination'].isin(['Abandoned', 'Rule Infraction'])]
# Print the shape of the updated dataframe
print("Shape of the updated dataframe:", new_chess_data_df.shape)


# In[17]:


print(new_chess_data_df.describe())


# In[18]:


print("Number of rows with '*' in column 'Result': ", (new_chess_data_df['Result'] == '*').sum())


# In[19]:


new_chess_data_df.head(5)


# In[20]:


eco_mapping = new_chess_data_df[['ECO', 'Opening']].drop_duplicates(subset='ECO')
eco_mapping


# In[21]:


eco_mapping.to_csv('eco_opening_mapping.csv', index=False)

# Print a message to indicate that the dataframe has been saved to a .csv file
print("Dataframe has been saved to 'eco_opening_mapping.csv'")


# In[22]:


new_chess_data_df.drop(['Opening'], axis=1, inplace=True)
new_chess_data_df.head(5)


# In[23]:


for col in new_chess_data_df.columns:
    if(len(new_chess_data_df[col].unique())<=20):
        print("Dealing with:", col)
        print(new_chess_data_df[col].unique())


# In[25]:


# Dealing with ECO values
# considering top 25 percentile White and Black ELO rated players
high_rated_whites = new_chess_data_df[new_chess_data_df["WhiteElo"] >= 1919]
high_rated_blacks = new_chess_data_df[new_chess_data_df["BlackElo"] >= 1919]
print("Total number of rows with top 25 percentile White and Black ELO rated players: ", (high_rated_whites.shape[0]+high_rated_blacks.shape[0]))

# Dealing with ECO values
# considering bottom 25 percentile White and Black ELO rated players
low_rated_whites = new_chess_data_df[new_chess_data_df["WhiteElo"] < 1559]
low_rated_blacks = new_chess_data_df[new_chess_data_df["BlackElo"] < 1557]
print("Total number of rows with bottom 25 percentile White and Black ELO rated players: ", (low_rated_whites.shape[0]+low_rated_blacks.shape[0]))

# Dealing with ECO values
# considering average rated White and Black ELO players
avg_rated_whites = new_chess_data_df[(new_chess_data_df["WhiteElo"] >= 1559) & (new_chess_data_df["WhiteElo"] < 1919)]
avg_rated_blacks = new_chess_data_df[(new_chess_data_df["BlackElo"] >= 1557) & (new_chess_data_df["BlackElo"] < 1919)]
print("Total number of rows with average White and Black ELO rated players: ", (avg_rated_whites.shape[0]+avg_rated_blacks.shape[0]))


# In[26]:


print(new_chess_data_df)


# In[28]:


import os
os.chdir(r'/kaggle/working')


# In[29]:


new_chess_data_df.to_csv(r'new_chess_data_df.csv')


# In[8]:


chess_data_df = pd.read_csv("chess_games.csv")
chess_data_df.head()


# In[9]:


chess_data_df.head()


# In[10]:


chess_data_df.shape


# In[3]:


chess_data_df_prep = pd.read_csv("/kaggle/input/dataset/chessdbpp.csv")
chess_data_df_prep.head()


# In[4]:


chess_data_df_prep.shape


# In[5]:


X = chess_data_df_prep.drop("Result", axis=1)
y = chess_data_df_prep["Result"]
cat_attributes = ['w1','b1','w2','b2','w3','b3']


# In[6]:


X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,y, train_size=100000, stratify=y,test_size=20000, random_state=42)

# print(X_Train.shape, X_Test.shape)
X_Train, X_Valid, Y_Train, Y_Valid = train_test_split(X,y, train_size=100000, stratify=y,test_size=10000, random_state=42)


# In[7]:


X_Train.shape, X_Test.shape, X_Valid.shape


# In[8]:


# define undersample strategy
random_undersampler = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
X_under, Y_under = random_undersampler.fit_resample(X_Train, Y_Train)


# In[9]:


X_under.head()


# In[11]:


X_under.shape, Y_under.shape


# In[15]:


num_attributes = ['WhiteElo','BlackElo']
cat_attributes = ['w1','b1','w2','b2','w3','b3']
print(num_attributes)
print(cat_attributes)


# In[13]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values


# In[16]:


num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attributes)),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attributes)),
        # ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(sparse=False, handle_unknown="ignore"))
    ])

data_prep_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[210]:


X_Test.head()


# In[201]:


# le = LabelEncoder()
# T_train = le.fit_transform(X_under)

# create an instance of the decision tree classifier
clf = DecisionTreeClassifier()

# fit the model
clf.fit(X_Train, Y_Train)

# make predictions on the test set
y_pred = clf.predict(X_Test)


# In[17]:


expLog = pd.DataFrame()
del expLog
try:
    expLog
except NameError:
    expLog = pd.DataFrame(columns=[ 
                                   "Model name",
                                   "Train Acc", 
                                   "Valid Acc",
                                   "Test  Acc",
                                   "Train AUC", 
                                   "Valid AUC",
                                   "Test  AUC",
                                   "Train F1", 
                                   "Valid F1",
                                   "Test F1",
                                   "Precision",
                                   "Recall",
                                   "Fit Time (seconds)"
                                  ])
expLog


# In[18]:


np.random.seed(42)
def run_model(model, X_train, Y_train, X_test, Y_test, X_valid, Y_valid):
    start_time = time.time()
    full_pipeline_with_predictor = Pipeline([
            ("preparation", data_prep_pipeline),
#             ("pca", PCA(n_components=60)),
            ("model", model)
        ])
    model = full_pipeline_with_predictor.fit(X_train, Y_train)
    model_name = "{}".format(type(full_pipeline_with_predictor['model']).__name__)
    fit_time = time.time() - start_time
    expLog.loc[len(expLog)] = [model_name] + list(np.round(
                   [accuracy_score(Y_train, model.predict(X_train)), 
                    accuracy_score(Y_valid, model.predict(X_valid)),
                    accuracy_score(Y_test, model.predict(X_test)),
                    roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1]),
                    roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1]),
                    roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1]),
                    f1_score(Y_train, model.predict(X_train)), 
                    f1_score(Y_valid, model.predict(X_valid)),
                    f1_score(Y_test, model.predict(X_test)),
                    precision_score(Y_test, model.predict(X_test)),
                    recall_score(Y_test, model.predict(X_test)),
                    fit_time], 4))


# In[19]:


clfs = [cb.CatBoostClassifier(),
        DecisionTreeClassifier(),
        GaussianNB(),RandomForestClassifier(),GradientBoostingClassifier(),XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),AdaBoostClassifier(random_state=42)]
for cl in clfs:
    print(cl)
    run_model(cl, X_Train, Y_Train, X_Test, Y_Test, X_Valid, Y_Valid)


# In[20]:


expLog


# In[28]:


def run_model_SVC(model, X_train, Y_train, X_test, Y_test, X_valid, Y_valid):
    start_time = time.time()
    full_pipeline_with_predictor = Pipeline([
            ("preparation", data_prep_pipeline),
            RBFSampler(gamma=1, random_state=42),
            ("model", model)
        ])
    model = full_pipeline_with_predictor.fit(X_train, Y_train)
    model_name = "{}".format(type(full_pipeline_with_predictor['model']).__name__)
    fit_time = time.time() - start_time
    expLog.loc[len(expLog)] = [model_name] + list(np.round(
                   [accuracy_score(Y_train, model.predict(X_train)), 
                    accuracy_score(Y_valid, model.predict(X_valid)),
                    accuracy_score(Y_test, model.predict(X_test)),
                    roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1]),
                    roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1]),
                    roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1]),
                    f1_score(Y_train, model.predict(X_train)), 
                    f1_score(Y_valid, model.predict(X_valid)),
                    f1_score(Y_test, model.predict(X_test)),
                    precision_score(Y_test, model.predict(X_test)),
                    recall_score(Y_test, model.predict(X_test)),
                    fit_time], 4))


# In[28]:


print(num_attributes)
print(cat_attributes)
# Create a StandardScaler object
scaler = StandardScaler()

# Create a OneHotEncoder object
encoder = OneHotEncoder(handle_unknown="ignore")

# Fit the StandardScaler object to the numerical data
X_Train_fitted = scaler.fit(X_Train[num_attributes])

# Transform the numerical data using the StandardScaler object
X_train_scaled = scaler.transform(X_Train_fitted)

# Fit the OneHotEncoder object to the categorical data
X_train_categorical = encoder.fit(X_Train[cat_attributes])

# Transform the categorical data using the OneHotEncoder object
X_train_categorical_encoded = encoder.transform(X_train_categorical)

# Concatenate the scaled numerical data and the encoded categorical data
X_Train = np.concatenate([X_train_scaled, X_train_categorical_encoded], axis=1)


# In[ ]:


# Fit the StandardScaler object to the numerical data
X_Test_fitted = scaler.fit(X_Test[num_attributes])

# Transform the numerical data using the StandardScaler object
X_test_scaled = scaler.transform(X_Test_fitted)

# Fit the OneHotEncoder object to the categorical data
X_test_categorical = encoder.fit(X_Train[cat_attributes])

# Transform the categorical data using the OneHotEncoder object
X_test_categorical_encoded = encoder.transform(X_test_categorical)

# Concatenate the scaled numerical data and the encoded categorical data
X_Test = np.concatenate([X_test_scaled, X_test_categorical_encoded], axis=1)


# In[26]:


if tf.test.is_gpu_available():
    # Use GPU
    device = "/gpu:0"
else:
    # Use CPU
    device = "/cpu:0"

# Create the model
model = Sequential()
model.add(Dense(64, input_dim=X_Train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_Train, Y_Train, epochs=100, batch_size=32, validation_data=(X_Valid, Y_Valid))

# Predict the classes for the test set
y_pred = model.predict_classes(X_Test)

# Evaluate the performance using accuracy, F1, precision, and recall metrics
accuracy = accuracy_score(Y_Test, y_pred)
f1 = f1_score(Y_Test, y_pred)
precision = precision_score(Y_Test, y_pred)
recall = recall_score(Y_Test, y_pred)

print("Accuracy: {:.2f}".format(accuracy))
print("F1 score: {:.2f}".format(f1))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))


# In[44]:


expLog


# In[ ]:


model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

