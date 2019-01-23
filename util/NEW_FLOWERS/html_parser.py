

file = open("81Flores.html")

lines = file.readlines()

for line in lines:
    if line.startswith('<h2>'):
        if line.__contains__('<i>'):
            print(line.split('<i>')[1].split('</i>')[0].replace(')', ''))
            # print(line.split('<i>')[0].split('<h2>')[1] .replace('(', ''))
        else:
            print(line.split('</i>')[0].replace(')', ''))
