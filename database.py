import pandas as pd
import glob

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
data = pd.read_csv('csv/angiospermsdatabase.csv', encoding='latin1', sep=';')

folders = glob.glob('dataset/78_classes/*')
cont = 0
info_df = pd.DataFrame()
not_found = list()
for folder in folders:
    name = folder.split('/')[-1].replace('_', ' ')
    # print(name)
    gender = name.split(' ')[0]
    species = name.split(' ')[-1]
    data_filtered = data[(data['Gênero'] == gender) & (data['Espécie'] == species)
                         & (data['Rank'] == 'Espécie')]
    data_filtered.insert(0, "latin_name", gender + ' ' + species)
    data_filtered.insert(1, "pop_name_pt_br", None)
    # print(data_filtered)
    if data_filtered.empty:
        cont += 1
        not_found.append(name)
    else:
        print(gender, species, len(data_filtered))
        # print(data_filtered)
        info_df = info_df.append(data_filtered)

for name in not_found:
    info_df = info_df.append(({'latin_name': name}), ignore_index=True)

print(info_df)
info_df = info_df.sort_values(by='latin_name')
info_df.to_json('info_df.json', orient='records', force_ascii=False, lines=True)
info_df.to_csv('info_df.csv')

print("Empty:", cont)
print("Total:", len(folders))
# print(folders)
