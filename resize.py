from PIL import Image
import os
import sys
import glob

root_dir = glob.glob('81_flowers/*')

for child_dir in root_dir:
    print(child_dir)
    for file_name in os.listdir(child_dir):
        # print(file_name)
        print("Processing %s %s" % (child_dir, file_name))
        image = Image.open(os.path.join(child_dir, file_name))
        # x, y = image.size
        new_dimensions = (512, 512)
        output = image.resize(new_dimensions, Image.ANTIALIAS)
        output_file_name = os.path.join(child_dir, file_name)
        output.save(output_file_name, "JPEG", quality=95)
print("\n------- All done ------- ")

