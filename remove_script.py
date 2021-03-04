import os
from skimage.io import imread

print('Removing inappropriate images...')
print('Images that have been removed:')
for (dirpath, dirname, filenames) in os.walk('dataset2'):
    for filename in filenames:
        if filename.endswith('jpg'):
            label = ''
            for i in filename:
                if i.isalpha():
                    label+=i
                else:
                    break
            filepath = os.path.join(dirpath, filename)
            image = imread(filepath)
            if len(image.shape)!=3 or image.shape[2]!=3 or label not in ['sunrise', 'rain', 'cloudy', 'shine']:
                print(filepath)
                os.remove(filepath)