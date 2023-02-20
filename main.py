#%% 1: Data Loading

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,r2_score
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model

#%% 2: 
CSV_PATH = os.path.join(os.getcwd(),'dataset', 'cases_malaysia_train.csv')
df = pd.read_csv(CSV_PATH)

#%% EDA
df.head()
df.info()
df.describe().T

# %% Data Cleaning
df['cases_new'] = pd.to_numeric(df['cases_new'],errors='coerce')
df.info()

plt.figure()
plt.plot(df['cases_new'])
plt.show()

df.isna().sum()

#%%
# interpolation
df['cases_new'] = df['cases_new'].interpolate(method='polynomial',order=2)

plt.figure()
plt.plot(df['cases_new'])
plt.show()

# %% Features selection
data = df['cases_new'].values

# %% Data Preprocessing

mms = MinMaxScaler()
data = mms.fit_transform(np.expand_dims(data,axis=-1))

X = []
y = []

win_size = 30

for i in range(win_size,len(data)):
     X.append(data[i-win_size:i])
     y.append(data[i])

#convert list to array
X = np.array(X)
y = np.array(y)


# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,shuffle=True,random_state=123)

# %% Model Develpment
input_shape = np.shape(X)[1:]

model = Sequential()
model.add(LSTM(64, input_shape=input_shape,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1,activation='relu'))
model.summary()

model.compile(optimizer='adam',loss='mse',metrics='mse')

plot_model(model,show_shapes=True)

#%%tensorboard

log_dir = os.path.join(os.getcwd(),datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb_callback = TensorBoard(log_dir=log_dir)

#early stopping
es_callback = EarlyStopping(monitor='val_mape', baseline=0.09, mode=min, patience=30, restore_best_weights =True)
hist = model.fit(X_train,y_train,epochs=500,validation_data=(X_test,y_test),callbacks=[tb_callback,es_callback])


# %% Model Analysis

print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()


# %% model comparison

y_true = y_test
y_pred = model.predict(X_test)

print('DL MAE error is {}'.format(mean_absolute_error(y_true,y_pred)))
print('DL MAPE error is {}'.format(mean_absolute_percentage_error(y_true,y_pred)))
print('DL R2 value is {}'.format(r2_score(y_true,y_pred)))

# %%
# Model Analysis against actual data

Test_csv = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
df_test = pd.read_csv(Test_csv)
df_test.info()
df_test['cases_new'] = df_test['cases_new'].interpolate(method='polynomial',order=2)


df_tot = pd.concat((df,df_test))
#mms
df_tot = mms.transform(np.expand_dims(df_tot['cases_new'],axis=-1))

X_actual = []
y_actual = []
for i in range(len(df),len(df_tot)):
     X_actual.append(df_tot[i-win_size:i])
     y_actual.append(df_tot[i])

X_actual = np.array(X_actual)
y_actual = np.array(y_actual)

#%% test the model

y_pred = model.predict(X_actual)

plt.figure()
plt.plot(y_pred,color='blue')
plt.plot(y_actual,color='red')
plt.legend(['Predicted cases', 'Actual cases'])
plt.show()

print('Actual DL MAE error is {}'.format(mean_absolute_error(y_actual,y_pred)))
print('Actual DL MAPE error is {}'.format(mean_absolute_percentage_error(y_actual,y_pred)))
print('Actual R2 value is {}'.format(r2_score(y_actual,y_pred)))

#%% Model Deployment
model.save('model.h5')

with open('mms.pkl','wb')as f:
    pickle.dump(mms,f)

#%%
