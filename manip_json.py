import pandas as pd

json = pd.read_json('flowers.json')

print(json[['popularNames', 'latinName', 'grupo']])

json.to_csv('flow.csv', index=False)

