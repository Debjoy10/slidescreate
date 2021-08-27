import numpy as np
import cv2 as cv
from tqdm import tqdm
from PIL import Image
import os
import glob
import shutil

def get_matches(query_img, train_img, frame_id):   
    # Convert it to grayscale
    img1 = cv.cvtColor(query_img,cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)
    
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    count = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            count += 1

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    
    # Show the final image
    if count / len(matches) < 0.4 or len(matches) == 0:
        strid = "{:09d}".format(frame_id)
        cv.imwrite("slides/Matches_{}.jpg".format(strid), train_img)

def main(video_filename, pdf_filename):
    # Video Capture
    cap = cv.VideoCapture(video_filename)
    if pdf_filename is None:
        pdf_filename = video_filename.split('/')[-1].split('.')[0]+'.pdf'

    # Init params
    if not os.path.exists('slides'):
        os.mkdir('slides')
    else:
        for f in os.listdir('slides'):
            os.remove(os.path.join('slides', f))
    pre_frame = None
    frame_id = 0
    frame_skip = 100
    nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)
    limperc = 1
    diffperc = 1

    print("Starting PDF convert process...")
    while(cap.isOpened()):
        if 100*(frame_id/nframes) > limperc:
            print("{} percent done". format(limperc))
            limperc += diffperc

        frame_id += 1
        try:
            ret, frame = cap.read()
            if pre_frame is None:
                pre_frame = frame
                # Start Frames
                strid = "{:09d}".format(frame_id)
                cv.imwrite("slides/Matches_{}.jpg".format(strid), pre_frame)

            if frame_id % frame_skip == 0:
                get_matches(pre_frame, frame, frame_id)
                pre_frame = frame
        except:
            break

    try:
        # Final Frame
        strid = "{:09d}".format(frame_id)
        cv.imwrite("slides/Matches_{}.jpg".format(strid), pre_frame)
        cap.release()
    except:
        pass

    # Converting to pdf
    print("Now Converting to pdf ....")
    imagelist = [os.path.join('slides', f) for f in os.listdir('slides')]
    imagelist.sort()
    ims = [Image.open(impath) for impath in imagelist]
    if len(ims) > 0:
        im1 = ims[0]
        ims = ims[1:]
        im1.save(pdf_filename, "PDF" ,resolution=100.0, save_all=True, append_images=ims)
        print("Done")
    else:
        raise Exception("Video Empty, Aborted")

    # Deleting slide pics
    shutil.rmtree('slides')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--videofile", help = "Lecture Video Filepath", required=True)
parser.add_argument("-p", "--pdffile", help = "Write PDF Filepath")

# Read arguments from command line
args = parser.parse_args()
main(args.videofile, args.pdffile)