import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

data = pd.read_csv('angiospermsdatabase.csv', encoding='latin1', sep=';')

data = data[data['Espécie'] == 'oleander']

print(data.head())

