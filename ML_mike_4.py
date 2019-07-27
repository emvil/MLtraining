# [70] Working with ordinal and categorical data
# https://pythonhealthcare.org/2018/04/17/70-machine-learning-working-with-ordinal-and-categorical-data/

import pandas as pd

# Example 1 : ordinal data
colour = ['green', 'green', 'red', 'blue', 'green', 'red','red']
size = ['small', 'small', 'large', 'medium', 'medium','x large', 'x small']
df = pd.DataFrame()
df['colour'] = colour
df['size'] = size
print(df)

# Define mapping dictionary:
size_classes = {'x small': 1,
                'small': 2,
                'medium': 3,
                'large': 4,
                'x large': 5}
# Map to dataframe and put the results into a new column:
df['size_number'] = df['size'].map(size_classes)
print(df)

# Example 2: categorical data
colours_df = pd.get_dummies(df['colour'])
print(colours_df)

# concatenate
df = pd.concat([df, colours_df], axis=1, join='inner')
del colours_df
print(df)

# select the categorical columns
df1 = df.loc[:, 'size_number':]
print(df1)
