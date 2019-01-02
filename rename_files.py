import os
import shutil

dirs = next(os.walk('../NEW_FLOWERS/'))[1]
print(dirs)
print(len(dirs))

for dir in dirs:
    os.system('cd ' + dir.replace(' ', '\ ') + '/ &&  ls | cat -n | while read n f; do mv "$f" "$n.jpg"; done ')

