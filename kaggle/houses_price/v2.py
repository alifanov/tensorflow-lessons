import csv
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

TARGET_COLUMN = 'SalePrice'

data_test = pd.read_csv('./test.csv')


def prepare_data():
    data = pd.read_csv('./train.csv')

    categorical_fields = data.select_dtypes(exclude=[np.number]).columns

    num_fields = data.select_dtypes(include=[np.number]).drop(TARGET_COLUMN, 1).columns

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

    data[TARGET_COLUMN].fillna(data[TARGET_COLUMN].mean(), inplace=True)

    X = data.values[:, 1:80]
    y = data.values[:, 80]
    X_test = data_test.values[:, 1:]

    return X, y, X_test


def create_model(
        n_input,
        init_mode='uniform',
        activation='relu'
):
    model = Sequential()

    model.add(Dense(n_input, input_dim=n_input, activation=activation, kernel_initializer=init_mode))
    model.add(Dense(64, activation=activation, kernel_initializer=init_mode))
    model.add(Dense(1))

    learning_rate = 1e-2
    decay = learning_rate / 1000

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=decay))
    return model


X, y, X_test = prepare_data()
X_train = X
y_train = y

n_input = X.shape[1]

# x_scaler = preprocessing.MinMaxScaler()
# X_scaled = x_scaler.fit_transform(X)
# X_test = x_scaler.fit_transform(X_test)
# X_train = X_scaled


nb_epoch = 5000
np.random.seed(3)
model = KerasRegressor(build_fn=create_model, n_input=n_input, epochs=nb_epoch, batch_size=5, verbose=1)
model.fit(X_train, y)

y_pred = model.predict(X_test)
writer = csv.writer(open('submission.csv', 'w'))
writer.writerow(['Id', 'SalePrice'])
for id, y in zip(data_test['Id'], y_pred):
    writer.writerow([id, y])
