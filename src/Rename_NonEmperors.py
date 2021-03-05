import os
import glob as glob


EXTENSION = '.jpg'
i = 0
for images_name in os.listdir("."):
    if not images_name.split('.')[1] == 'py':
        rename_to = str(i) + EXTENSION
        os.rename(images_name, rename_to)
        i+=1