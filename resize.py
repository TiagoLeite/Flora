from PIL import Image
import os
import sys
import glob
import random
import os

MAX = 99999999

save_dir = 'image_android/'

root_dir = glob.glob('dataset/78_classes/*')

for child_dir in root_dir:

    print('mkdir ' + 'image_android/' + child_dir.split('/')[-1].lower())
    os.system('mkdir ' + 'image_android/' + child_dir.split('/')[-1].lower())
    cont = 0
    for file_name in os.listdir(child_dir):
        # print(file_name)
        print("Processing %s %s" % (child_dir, file_name))
        image = Image.open(os.path.join(child_dir, file_name))
        # x, y = image.size
        new_dimensions = (200, 200)
        # output = image.crop(box=(50, 50, ))
        output = image.resize(new_dimensions, Image.ANTIALIAS)
        output_file_name = os.path.join('image_android/' + child_dir.split('/')[-1].lower(), str(cont)+'.jpg')
        print(output_file_name)

        try:
            output.save(output_file_name, "JPEG", quality=75, optimize=True)
            cont += 1

        except OSError:
            continue

        if cont >= MAX:
            break

print("\n------- All done ------- ")

