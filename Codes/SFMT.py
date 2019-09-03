import cv2
from imutils import is_cv2
import multiprocessing
from timeit import default_timer

#opening file to write time taken to process frames
f=open('SFMT.txt','w')
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
def process(image,stri):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        thresh=cv2.threshold(gray,60,255,cv2.THRESH_BINARY)[1]
        cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts=cnts[0] if is_cv2() else cnts[1]
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
                cv2.imshow(stri,image)

#loop to call process() function and write time taken to process frames in file
#loop contains 4 multithread process, each processing a quadrant of the frame
#To get best analysis results, time taken process 4 frames is averaged out before writing to file
if __name__=='__main__':
        while default_timer()<=10:
            ret_val,img=cam.read()
            start = default_timer()
            h,w,d=img.shape
            p1=multiprocessing.Process(target=process,args=(img[:int(h/2),:int(w/2)],"top-left",))
            p2=multiprocessing.Process(target=process,args=(img[int(h/2):,:int(w/2)],"bottom-left",))
            p3=multiprocessing.Process(target=process,args=(img[:int(h/2),int(w/2):],"top-right",))
            p4=multiprocessing.Process(target=process,args=(img[int(h/2):,int(w/2):],"bottom-right",))
            p1.run()
            p2.run()
            p3.run()
            p4.run()
            stop = default_timer()
            if ctr==3:l,ctr=[],0
            l.append(stop-start)
            f.writelines(str(sum(l)/4)+'\n')
            ctr=ctr+1
            if cv2.waitKey(1)==27:break
        cv2.destroyAllWindows()
        f.flush()
        f.close()
