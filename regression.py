import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data using pandas
# read_fwf - pandas dataframe object
dataframe = pd.read_fwf('brain_body.txt')

# parse and read both measurements into two seperate variables

# brain weight
x_values = dataframe[["Brain"]]
# body weight
y_values = dataframe[["Body"]]

# train model on data
# use scikit learn's linear model object to initialize
# linear regression and store it in the body regression
# variable
body_reg = linear_model.LinearRegression()
# fit our model on x, y value pairs
body_reg.fit(x_values, y_values)

# visualize results
plt.scatter(x_values, y_values)
# plot our regression line
# for every x value predict the associated y value
# and draw a line that intersects all those points
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
