import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn import cross_validation

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from matplotlib import pyplot

data = pd.read_csv('./train.csv')
print('Data shape: {}'.format(data.shape))

data_test = pd.read_csv('./test.csv')
TARGET_COLUMN = 'SalePrice'

categorical_fields = data.select_dtypes(exclude=[np.number]).columns
# print(categorical_fields)

num_fields = data.select_dtypes(include=[np.number]).drop(TARGET_COLUMN, 1).columns
# print(num_fields)

for col in categorical_fields:
    data[col].fillna('default', inplace=True)
    data_test[col].fillna('default', inplace=True)

for col in num_fields:
    data[col].fillna(0, inplace=True)
    data_test[col].fillna(0, inplace=True)

encode = preprocessing.LabelEncoder()
for col in categorical_fields:
    data[col] = encode.fit_transform(data[col])
    data_test[col] = encode.fit_transform(data_test[col])

data[TARGET_COLUMN].fillna(0, inplace=True)

y = data[TARGET_COLUMN].as_matrix()
X = data.drop(TARGET_COLUMN, 1)
X = X.drop('Id', 1).as_matrix()

n_input = X.shape[1]

x_scaler = preprocessing.MinMaxScaler()
y_scaler = preprocessing.MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X_scaled, y_scaled, test_size=0.1, random_state=3)

# print('X_train: {}'.format(X_train.shape))
# print('X_test: {}'.format(X_test.shape))
# print('y_train: {}'.format(y_train.shape))
# print('y_test: {}'.format(y_test.shape))

model = Sequential()

model.add(Dense(1024, input_dim=n_input, activation='relu'))
# model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(1))

learning_rate = 1e-4
n_epochs = 1000
decay = learning_rate / n_epochs

model.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=['accuracy'], decay=decay)
np.random.seed(3)
model.fit(X_train, y_train, epochs=n_epochs, batch_size=10, verbose=1)

# predicted = model.predict(X_test)

# pyplot.plot(y_scaler.inverse_transform(predicted), color="blue")
# pyplot.plot(y_scaler.inverse_transform(y_test), color="green")
# pyplot.show()



X_test = data_test.drop('Id', 1)
X_test_scaled = x_scaler.fit_transform(X_test)
print(X_test_scaled.shape)

y_pred = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred)
print(y_pred.shape)

i = 0
file = open('submission.csv', 'w')
header = "Id,SalePrice"
header = header + '\n'
file.write(header)
for id in (data_test['Id']):
    str = "{},{}".format(id, y_pred[i][0])
    str = str + '\n'
    file.write(str)
    i += 1
