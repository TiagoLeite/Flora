import os
import shutil

files_to_find = []

dirs = next(os.walk('../NEW_FLOWERS/downloads'))[1]

print(dirs)
print(len(dirs))

for k in range(len(dirs)):

    if k % 2 == 1:
        files = os.listdir('../NEW_FLOWERS/downloads/' + dirs[k])
        if os.path.isfile('../NEW_FLOWERS/downloads/' + dirs[k] + '/Thumbs.db'):
            os.remove('../NEW_FLOWERS/downloads/' + dirs[k] + '/Thumbs.db')
        if os.path.isfile('../NEW_FLOWERS/downloads/' + dirs[k - 1] + '/Thumbs.db'):
            os.remove('../NEW_FLOWERS/downloads/' + dirs[k - 1] + '/Thumbs.db')
        for f in files:
            # print(f)
            try:
                print("Moved ", f)
                if os.path.isfile('../NEW_FLOWERS/downloads/' + dirs[k - 1] + '/'+f):
                    os.remove('../NEW_FLOWERS/downloads/' + dirs[k - 1] + '/'+f)
                shutil.move('../NEW_FLOWERS/downloads/' + dirs[k] + '/' + f,
                            '../NEW_FLOWERS/downloads/' + dirs[k - 1])
            except FileNotFoundError as e:
                continue
                # if os.path.isfile('../NEW_FLOWERS/downloads/' + dirs[k] + '/' + f):
                #    os.remove('../NEW_FLOWERS/downloads/' + dirs[k] + '/' + f),

        shutil.rmtree('../NEW_FLOWERS/downloads/' + dirs[k])
