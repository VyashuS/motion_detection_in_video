import cv2
import numpy as np


# input = 'ishacenter.mp4'
input = 0
source = input

videocapture = cv2.VideoCapture(source)

if videocapture.isOpened() is False:
    print("Error opening video stream or file")

frame_h = int(videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_fps = int(videocapture.get(cv2.CAP_PROP_FPS))

size = (frame_w,frame_h)
size_quad = (int(frame_w*2),int(frame_h*2))

video_quad = cv2.VideoWriter('videoquad.mp4',cv2.VideoWriter_fourcc(*'mp4v'), frame_fps, size_quad)

def banner(frame, text,banner_h=0.08, text_color = (0,255,0),font_scale = 1):
    height = int(banner_h * frame.shape[0])
    cv2.rectangle(frame, (0,0),(frame.shape[1],height), (0,0,0), thickness = -1)

    left = 30
    location = (left,int(10+(height/2)))
    cv2.putText(frame,text,location,cv2.FONT_HERSHEY_SIMPLEX,font_scale,text_color,thickness = 2,lineType=cv2.LINE_AA)


bg_sub = cv2.createBackgroundSubtractorKNN(history=200)

ksize = (5,5)
red = (255,0,0)
yellow= (255,255,0)

count = 0


while True:
    ret, frame = videocapture.read()

    if frame is None:
        break
    else:
        frame_erode = frame.copy()

    # step 1
    fg_mask = bg_sub.apply(frame)
    motion = cv2.findNonZero(fg_mask)
    x, y, w, h = cv2.boundingRect(motion)

    # step2
    fg_mask_erode = cv2.erode(fg_mask,np.ones(ksize,np.uint8))
    motion_erode = cv2.findNonZero(fg_mask_erode)
    xe,ye,we,he = cv2.boundingRect(motion_erode)

    if motion is not None:
        cv2.rectangle(frame,(x,y),(x+w,y+h),red,thickness = 3,lineType=cv2.LINE_8)

    if motion_erode is not None:
        cv2.rectangle(frame_erode,(xe,ye),(xe+we,ye+he),red,thickness = 3,lineType=cv2.LINE_8)

    bgr_fg_mask = cv2.cvtColor(fg_mask,cv2.COLOR_GRAY2BGR)
    bgr_fg_mask_erode = cv2.cvtColor(fg_mask_erode, cv2.COLOR_GRAY2BGR)

    count = count + 1
    banner(frame,'Frame No ; '+str(int(count)))
    banner(frame_erode, 'Frame No ; ' + str(int(count)))

    banner(bgr_fg_mask,'FORE GROUNG MASK')
    banner(bgr_fg_mask_erode,'FOREGROUNG MASK ERODE')

    # build a quad view
    top = np.hstack([bgr_fg_mask,frame])
    bot = np.hstack([bgr_fg_mask_erode,frame_erode])
    frame_compos = np.vstack([top,bot])

    fh,fw,_ = frame_compos.shape
    cv2.line(frame_compos,(0,int(fh/2)),(fw,int(fh/2)),yellow,thickness = 3,lineType=cv2.LINE_8)



    video_quad.write(frame_compos)
    cv2.imshow('Output',frame_compos)
    k = cv2.waitKey(9)
    if k == ord('q'):
        break

videocapture.release()
video_quad.release()
cv2.destroyAllWindows()


