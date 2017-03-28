import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn import cross_validation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from matplotlib import pyplot

TARGET_COLUMN = 'SalePrice'

data_test = pd.read_csv('./test.csv')


def prepare_data():
    data = pd.read_csv('./train.csv')

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

    X = data.drop(TARGET_COLUMN, 1)
    X = X.drop('Id', 1).as_matrix()
    y = data[TARGET_COLUMN].as_matrix()
    X_test = data_test.drop('Id', 1).as_matrix()

    return X, y, X_test


def create_model(
        n_input,
        init_mode='lecun_uniform',
        activation='softplus',
        n_hidden=100
):
    model = Sequential()

    model.add(Dense(n_hidden
    model.add(Dense(64, activation=activation, init=init_mode))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(1))

    learning_rate = 1e-1
    decay = 1e-3

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=decay))
    return model


X, y, X_test = prepare_data()

n_input = X.shape[1]

# x_scaler = preprocessing.MinMaxScaler()
# y_scaler = preprocessing.MinMaxScaler()
#
# X_scaled = x_scaler.fit_transform(X)
# y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(
#     X_scaled, y_scaled, test_size=0.1, random_state=3)

# print('X_train: {}'.format(X_train.shape))
# print('X_test: {}'.format(X_test.shape))
# print('y_train: {}'.format(y_train.shape))
# print('y_test: {}'.format(y_test.shape))

# n_epochs = 100
# model = create_model(n_epochs)
# np.random.seed(3)
# model.fit(X, y, epochs=n_epochs, batch_size=10, verbose=1)

# GridSearchCV
model = KerasRegressor(build_fn=create_model, n_input=n_input, nb_epoch=1000, batch_size=50, verbose=0)
param_grid = {
    'n_hidden': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, scoring='neg_mean_squared_error')
grid_result = grid.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# seed = 7
# np.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=create_model, n_input=n_input, n_epochs=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
#
# kfold = KFold(n_splits=3, random_state=seed)
# results = cross_val_score(pipeline, X, y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# predicted = model.predict(X_test)

# pyplot.plot(y_scaler.inverse_transform(predicted), color="blue")
# pyplot.plot(y_scaler.inverse_transform(y_test), color="green")
# pyplot.show()



# X_test_scaled = x_scaler.fit_transform(X_test)
# print(X_test_scaled.shape)
#
# y_pred = model.predict(X_test_scaled)
# y_pred = y_scaler.inverse_transform(y_pred)
# print(y_pred.shape)

# file = open('submission.csv', 'w')
# header = "Id,SalePrice\n"
# file.write(header)
# for id, y in zip(data_test['Id'], y_pred):
#     file.write('{},{}\n'.format(id, y[0]))
