import os
from skimage.io import imread
import io
import zipfile
import requests
import shutil
import zlib
import gzip

r = requests.get('https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/4drtyfjtfy-1.zip')
z = zipfile.ZipFile(io.BytesIO(r.content), 'r')
z.extractall()
z = zipfile.ZipFile('dataset2.zip', 'r')
z.extractall()

print('Removing inappropriate images...')
print('Images that have been removed:')
for (dirpath, dirname, filenames) in os.walk('dataset2'):
    for filename in filenames:
        label = ''
        for i in filename:
            if i.isalpha():
                label+=i
            else:
                break
        filepath = os.path.join(dirpath, filename)
        image = imread(filepath)
        if not (filename.endswith('.jpg') or filename.endswith('.jpeg'))  or len(image.shape)!=3 or \
                image.shape[2]!=3 or label not in ['sunrise', 'rain', 'cloudy', 'shine']:
            print(filepath)
            os.remove(filepath)
