import cv2
import argparse
import imutils

#function to detect shapes and return text
def detect(c):
        shape = "unidentified"
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)
	if len(approx) == 3:shape = "triangle"
	elif len(approx) == 4:
		(x, y, w, h) = cv2.boundingRect(approx)
		ar = w / float(h)
		shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
	elif len(approx) == 5:shape = "pentagon"
	else:shape = "circle"
	return shape

#Single Core Processing code
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the input image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
for c in cnts:
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]))
	cY = int((M["m01"] / M["m00"]))
	shape = detect(c)
	c = c.astype("float")
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
