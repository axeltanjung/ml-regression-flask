import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Get path to file dynamically
WD = os.path.dirname(os.path.abspath(__file__))

# read data source
dataset = pd.read_csv('data/salary.csv')

# define x (independent) and y (dependent) variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split data to train (67%) and test (33%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Develop regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# save model to pickle
pickle.dump(regressor, open(WD + '/pickled/model.pkl','wb'))


# load pickled model
model = pickle.load(open(WD + '/pickled/model.pkl','rb'))
print(model.predict([[1.8]]))
