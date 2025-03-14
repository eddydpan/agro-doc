import pyapriltags
import cv2
import numpy as np
import math


at_detector = pyapriltags.Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

img = cv2.imread("/home/eddy/github.com/agro-doc/imseg/at_bench_test.jpeg")
img = cv2.resize(img, (900, 675))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.shape)
# print(gray.shape)

# cv2.imshow("img", img)
# cv2.imshow("gray", gray)
tags = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None) # might need to change estimate_tag_pose to True
print(f"Tags is of size {len(tags)}\n. Tags[0] looks like: {type(tags[0].corners)}\n and Tags has the following values: {tags}")
image_with_rectangles = img.copy()
# image_with_rectangles = cv2.cvtColor(image_with_rectangles, cv2.COLOR_GRAY2BGR)
bboxes = {}
for tag in tags:
    
    # tag.corners is a nd.array
    x, y = gray.shape
    w, h = (0, 0)
    for i in range(len(tag.corners)):
        x, y = int(min(x, tag.corners[i][0])), int(min(y, tag.corners[i][1]))
        w, h = int(max(w, tag.corners[i][0])), int(max(h, tag.corners[i][1]))
    print(f"x,y: ({x}, {y})")
    print(f"w,h: ({w}, {h})")
    cv2.rectangle(image_with_rectangles, (x, y), (w, h), (0, 255, 0), 2)
    bboxes[tag.tag_id] = (x,w,y,)

# Draw the inside line, prioritize right and bottom edges
start = (bboxes[0][1], bboxes[0][2])
end = (bboxes[1][1], bboxes[1][2])
print(start, end)
cv2.line(image_with_rectangles, start, end, (255, 0, 0), 2)
cv2.imshow("bounding boxes", image_with_rectangles)

cv2.waitKey(0)
