import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from git import Repo

from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

TARGET_COLUMN = 'SalePrice'
EPOCHS = 700
LR = 1e-3
BATCH_SIZE = 100

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

    # data[TARGET_COLUMN].fillna(data[TARGET_COLUMN].mean(), inplace=True)
    data.dropna(subset=[TARGET_COLUMN], inplace=True)

    X = data.values[:, 1:-1]
    y = data.values[:, -1]

    X_test = data_test.values[:, 1:]

    return X, y, X_test


def create_model(
        n_input,
        activation='relu',
        dropout=0.2
):
    model = Sequential()

    model.add(Dense(n_input, input_dim=n_input, activation=activation, kernel_initializer='uniform'))
    # model.add(Dropout(dropout))
    model.add(Dense(512, activation=activation, kernel_initializer='uniform'))
    # model.add(Dropout(dropout))
    model.add(Dense(256, activation=activation, kernel_initializer='uniform'))
    # model.add(Dropout(dropout))
    # model.add(Dense(128, activation=activation))
    # model.add(Dropout(dropout))
    # model.add(Dense(64, activation=activation))
    # model.add(Dropout(dropout))
    # model.add(Dense(32, activation=activation))
    # model.add(Dropout(dropout))
    # model.add(Dense(16, activation=activation))
    # model.add(Dropout(dropout))
    model.add(Dense(1, activation=activation, kernel_initializer='uniform'))

    learning_rate = LR
    decay = learning_rate / EPOCHS

    model.compile(loss='mse', optimizer=Adam(lr=LR, decay=decay))
    return model


X, y, X_validation = prepare_data()

n_input = X.shape[1]

nb_epoch = EPOCHS
np.random.seed(3)
model = KerasRegressor(build_fn=create_model, n_input=n_input, epochs=nb_epoch, batch_size=BATCH_SIZE, verbose=1)
history = model.fit(X, y, validation_split=0.33)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

rmse_test = history.history['val_loss'][-1]
print()
print('Test RMSE: {0:.4}'.format(rmse_test))

config = 'bs_{}-lr_{}-e_{}-rmse_{:.2f}'.format(
    BATCH_SIZE,
    LR,
    EPOCHS,
    rmse_test
)

repo = Repo('../..')
git = repo.git
git.commit('.', m='new attempt: {}'.format(config))
config += '.hash-{}'.format(repo.commit().hexsha)

y_pred = model.predict(X_validation)
writer = csv.writer(open('submission.{}.csv'.format(config), 'w'))
writer.writerow(['Id', 'SalePrice'])
for id, y in zip(data_test['Id'], y_pred):
    writer.writerow([id, y])

git.push('origin')