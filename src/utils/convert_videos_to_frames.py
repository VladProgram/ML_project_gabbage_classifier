import os
import cv2
import numpy as np

frame_rate = 10
output_folder = 'data/frames'
cwd = os.getcwd()


def get_video_files(folder):
    return os.listdir(folder)

def create_folder(folder):
    # create folder with video name
    path = os.path.join(cwd, output_folder, folder)
    if not os.path.exists(path):
        os.mkdir(path)

    return path

def capture_frames_from_video(video_name, output_folder):
    video_path = os.path.join(cwd,'data/video', video_name)
    capture = cv2.VideoCapture(video_path)
    # print(type(capture))
    # print(f'video_path: {video_path}')

    frameNr = 0
    while (True):
        frameNr += 1
        # print('frameNr', frameNr)
        success, frame = capture.read()

        if not success:
            break

        if frameNr % frame_rate == 0 or frameNr == 1:
            # print('writing fr', frameNr, frameNr % frame_rate)
            # cv2.imwrite(f'C:/Users/vlado/OneDrive/Desktop/Pic_zele/a_video_{frameNr}.jpg', frame)
            # print(output_folder)
            cv2.imwrite(f'{output_folder}/{frameNr}.jpg', frame)

    capture.release()
    print("-----------------------")

video_files = get_video_files('data/video')
for video_file in video_files:
    # video_folder = create_folder(video_file)
    capture_frames_from_video(video_file, output_folder)




# https://techtutorialsx.com/2021/04/29/python-opencv-splitting-video-frames/
