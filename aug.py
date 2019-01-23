import Augmentor
import shutil
import os
import glob


def augment(folder):
    p = Augmentor.Pipeline(folder)
    p.shear(probability=1.0, max_shear_right=25, max_shear_left=25)
    p.flip_left_right(probability=0.5)
    p.crop_random(probability=0.5, percentage_area=0.9)
    # p.crop_random(probability=1, percentage_area=0.9)
    # p.rotate(max_right_rotation=10, max_left_rotation=10, probability=0.75)
    p.resize(probability=1.0, width=224, height=224)
    p.set_seed(17)
    p.sample(100)


folders = glob.glob('60_flowers/*')
cont = 0
less = 10000000
for folder in folders:
    cont += 1
    list = os.listdir(folder)  # dir is your directory path
    number_files = len(list)
    augment(folder)
    files = glob.glob(folder + "/output/*.jpg")
    print(cont)
    for file in files:
        #print(file)
        shutil.move(file, folder)
    os.system('cd ' + folder + ' && rm -rf output/ && ls | cat -n | while read n f; do mv "$f" "$n.jpg"; done')

print('LEss:', less)
