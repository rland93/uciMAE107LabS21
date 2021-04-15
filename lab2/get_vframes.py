import cv2, os
from pathlib import Path
'''
Video Processing script

requires opencv-python

- look through directory of videos in `videos/`
- we assume that video framerate is 30 fps
- split into jpg frames at 5fps
- save jpg frames with filename of video plus timestamps
  into `frames/[video stem]/` folder
'''


VIDEO_FOLDER = 'videos'
FRAME_FOLDER = './frames'
for p in Path(VIDEO_FOLDER).iterdir():
    vname = str(p.stem)
    vidcap = cv2.VideoCapture(str(p))
    f = 0
    success = True
    # make new folder with video name
    foldername = FRAME_FOLDER + '/' + str(p.stem)
    os.makedirs(foldername, exist_ok=False)
    while success:
        success, image = vidcap.read()
        # we can reduce the number of images significantly without 
        # a major loss of fidelity (1/30th of a second -> 1/5th of a second)
        # with 5x less images to look through...
        if f % 6 == 0:
            t = str(f / 30).split('.')
            # write actual timestamp
            writepath = foldername + '/' + vname + '_' + str(t[0]).rjust(4, '0') + 's' + str(t[1]) + '00' + 'ms' + '.jpg'
            # save resized image
            cv2.imwrite(writepath, cv2.resize(image, (640, 360) ))
        f += 1