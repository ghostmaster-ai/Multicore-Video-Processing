import cv2
import imutils
import timeit

#opening file to write time taken to process frames
f=open('single_thread.txt','w')
l=[]
ctr=0

#function to detect shapes and return text
def detect(c):
        shape="unidentified"
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.04 * peri,True)
        if len(approx)==3:shape="triangle"
        elif len(approx)==4:
                (x,y,w,h)=cv2.boundingRect(approx)
                ar=w/float(h)
                shape="square" if ar>=0.95 and ar<=1.05 else "rectangle"
        elif len(approx)==5:shape="pentagon"
        else:shape="circle"
        return shape

#Capturing live Video Footage off of the Web Cam to recognise shapes 
cam=cv2.VideoCapture(0)

#single core processing code
def process(image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        thresh=cv2.threshold(gray,60,255,cv2.THRESH_BINARY)[1]
        cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts=cnts[0] if imutils.is_cv2() else cnts[1]
        for c in cnts:
                M=cv2.moments(c)
                cX,cY=0,0
                try:
                        cX=int((M["m10"]/M["m00"]))
                        cY=int((M["m01"]/M["m00"]))
                except:pass
                shape=detect(c)
                c=c.astype("float")
                c=c.astype("int")
                cv2.drawContours(image,[c],-1,(0,255,0),2)
                cv2.putText(image,shape,(cX,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                cv2.imshow("Image",image)

#loop to call process() function and write time taken to process frames in file
while timeit.default_timer()<=10:
    ret_val,img=cam.read()
    start=timeit.default_timer()
    process(img)
    stop=timeit.default_timer()
    if ctr==3:l,ctr=[],0
    l.append(stop-start)
    f.writelines(str(sum(l)/4)+'\n')
    ctr=ctr+1
    if cv2.waitKey(1)==27: break
cv2.destroyAllWindows()
f.flush()
f.close()
