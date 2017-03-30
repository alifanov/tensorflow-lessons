import csv
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

TARGET_COLUMN = 'SalePrice'
EPOCHS = 300
LR = 1e-2
BATCH_SIZE = 50

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
        activation='relu'
):
    model = Sequential()

    model.add(Dense(n_input, input_dim=n_input, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(1))

    learning_rate = LR
    decay = learning_rate / EPOCHS

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=decay))
    return model


X, y, X_validation = prepare_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train = X
# y_train = y

n_input = X.shape[1]

# x_scaler = preprocessing.MinMaxScaler()
# X_scaled = x_scaler.fit_transform(X)
# X_test = x_scaler.fit_transform(X_test)
# X_train = X_scaled


nb_epoch = EPOCHS
np.random.seed(3)
model = KerasRegressor(build_fn=create_model, n_input=n_input, epochs=nb_epoch, batch_size=BATCH_SIZE, verbose=1)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
rmse_test = mean_squared_error(y_test, y_test_pred)**0.5
print()
print('Test RMSE: {0:.4}'.format(rmse_test))

config = 'bs_{}-lr_{}-e_{}-rmse_{:.2f}'.format(
    BATCH_SIZE,
    LR,
    EPOCHS,
    rmse_test
)

y_pred = model.predict(X_validation)
writer = csv.writer(open('submission.{}.csv'.format(config), 'w'))
writer.writerow(['Id', 'SalePrice'])
for id, y in zip(data_test['Id'], y_pred):
    writer.writerow([id, y])
