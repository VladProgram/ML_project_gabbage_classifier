import os
counter = 0
files = []
file_names = os.listdir(r'data\frames')
for i in file_names:
    path = r'data\frames\{}'.format(i)
    files.append(path)
    # print(path)

for ix, file in enumerate(files):
    images = os.listdir(file)
    for img_n, img in enumerate(images):
        img_path = file + '\\' + img
        new_path = file + '\\' + r'{}_{}.jpg'.format(ix, counter)
        os.rename(img_path, new_path)
        counter += 1
        print(img_path, counter)
        
        
        