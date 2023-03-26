import json
import pandas as pd
import os

def convert_json(input_file, output_dir):
  with open(input_file, "r") as read_file:
    d = json.load(read_file)
  
  convertor = {'cabbage': 0, 'weed': 1}
  for key in d['frames'].keys():
    txt_name = key.replace('.jpg', '.txt')
    txt_list = []
    for i in d['frames'][key]:
        k = convertor[i['tags'][0]]
        w = (i['x2'] - i['x1']) / i['width']
        h = (i['y2'] - i['y1']) / i['height']
        xc = (i['x1'] / i['width']) + w/2
        yc = (i['y1'] / i['height']) + h/2
        row = (k, xc, yc, w, h)
        txt_list.append(row)
    columns = [ 'k', 'xc', 'yc', 'w', 'h']
    df = pd.DataFrame(txt_list, columns=columns)
    output_path = os.path.join(output_dir, txt_name)
    df.to_csv(output_path, sep='\t', index=False, header=False)
    print('wrote', output_path)

#convert_json(r'C:\ML_cource\Project\datasets_new\test\images.json', r'C:\ML_cource\Project\datasets_new\test\labels')
#convert_json(r'C:\ML_cource\Project\datasets_new\train\images.json', r'C:\ML_cource\Project\datasets_new\train\labels')
convert_json(r'C:\ML_cource\Project\datasets_new\val\images.json', r'C:\ML_cource\Project\datasets_new\val\labels')
