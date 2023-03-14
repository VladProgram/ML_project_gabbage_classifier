import os
counter = 0
folders = []
file_names = os.listdir('data/frames')
for i in file_names:
    path = f'data/frames/{i}'
    folders.append(path)
    # print(path)



for ix, folder_name in enumerate(folders):
    images = os.listdir(folder_name)
    for img_n, img in enumerate(images):
        # img_path = folder_name + '\\' + img
        img_path = os.path.join(folder_name, img)
        # new_path = folder_name + '\\' + f'{ix}_{counter}.jpg'
        new_path = os.path.join(folder_name, f'{ix}_{counter}.jpg')
        # os.rename(img_path, new_path)
        print(img_path, new_path)
        counter += 1



