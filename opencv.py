import cv2
import numpy as np
import imutils

from matplotlib import pyplot as plt



class FaceDetector:
    def __init__(self,cascadepath):
        self.cascade=cv2.CascadeClassifier(cascadepath)

    def detect_face(self,image):
        rects=self.cascade.detectMultiScale(image,scaleFactor=1.5,minNeighbors=5,minSize=(60,60),flags=cv2.CASCADE_SCALE_IMAGE)
        return rects


#image=cv2.imread("check.jpeg")
camera = cv2.VideoCapture(0)

while True:
    _,frame=camera.read()
    frame=imutils.resize(frame,width=300)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fd=FaceDetector("face.xml")

    rectan=fd.detect_face(gray)


    for (x,y,w,h) in rectan:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("face",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()



#O(nlogn)

##gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##blurred=cv2.GaussianBlur(gray,(5,5),0)
##thresh=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,4)
###cv2.imshow("thresh",thresh)
###(T,threshinv)=cv2.threshold(blurred,155,255,cv2.THRESH_BINARY_INV)
###cv2.imshow("inv",threshinv)
##
##cv2.imshow("final",thresh)









##image=cv2.imread("coins.jpg")
##gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##blur=cv2.GaussianBlur(gray,(17,17),0)
##
##canny=cv2.Canny(blur,150,210)
##cv2.imshow("canny",canny)
##
##(cnts,_)=cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
##print(len(cnts))
##
##
##cv2.drawContours(image,cnts,-1,(0,255,0),2)
##cv2.imshow("contours",image)
















































##gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##
##laplacian=cv2.Laplacian(gray,cv2.CV_64F)
##laplacian=np.uint8(np.absolute(laplacian))
##cv2.imshow("Laplacian",laplacian)
##
##sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0)
##sobely=cv2.Sobel(gray,cv2.CV_64F,0,1)
##
##
##sobelx=np.uint8(np.absolute(sobelx))
##sobely=np.uint8(np.absolute(sobely))
##
##cv2.imshow("sobelx",sobely)
##
##sobel=cv2.bitwise_or(sobelx,sobely)
##
##cv2.imshow("sobel",sobel)




















##gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##blurred=cv2.GaussianBlur(gray,(1,1),0)
##(T,thresh)=cv2.threshold(blurred,155,255,cv2.THRESH_BINARY)
###cv2.imshow("thresh",thresh)
##(T,threshinv)=cv2.threshold(blurred,155,255,cv2.THRESH_BINARY_INV)
###cv2.imshow("inv",threshinv)
##
##cv2.imshow("final",cv2.bitwise_and(image,image,mask=thresh))
##




##blurred=np.hstack([
##    cv2.bilateralFilter(image,7,51,51),
##    cv2.bilateralFilter(image,9,100,100)])
##
##cv2.imshow("changed",blurred)
cv2.waitKey(0)

#cv2.medainBlur(image,no)
#cv2.GaussianBlur(image,(no,no),0)
#cv2.blur(image,(no ,no))

#gray=cv2.cvtColor(image,cv2.COLOR_BGR2XYZ)
#cv2.imshow("gray",gray)
#hist=cv2.calcHist([image],[0],None,[256],[0,256])
##chans=cv2.split(image)
##colors=("b","g","r")
##plt.figure()
###plt.plot(hist,color='red')
##plt.xlim([0,256])
##
##
##for (chan,color) in zip(chans,colors):
##    hist=cv2.calcHist([chan],[0],None,[256],[0,256])
##    plt.plot(hist,color=color)
##    plt.xlim([0,256])
##plt.show()   

#cv2.calcHist(images,channels,mask,histSize,ranges)





##(B,G,R)=cv2.split(image)
##
####cv2.imshow("red",R)
####cv2.imshow("blue",B)
####cv2.imshow("grren",G)
##
##merge=cv2.merge([B,G,R])
##cv2.imshow("merged",merge)
##
##frame=np.zeros(image.shape[:2],dtype="uint8")
##cv2.imshow("red",cv2.merge([frame,G,frame]))




















##mask=np.zeros(image.shape[:2],dtype="uint8")
##(cX,cY)=(image.shape[1]//2,image.shape[0]//2)
##cv2.rectangle(mask,(cX-50,cY-50),(cX+50,cY+50),255,-1)
##cv2.imshow("Mask",mask)
##
##applymask=cv2.bitwise_and(image,image,mask=mask)
##cv2.imshow("masked",applymask)


















##frame=np.zeros((300,300),dtype="uint8")
##cv2.rectangle(frame,(25,25),(275,275),255,-1)
##cv2.imshow("rectangle",frame)
##
##frame1=np.zeros((300,300),dtype="uint8")
##cv2.circle(frame1,(150,150),150,255,-1)
##cv2.imshow("circle",frame1)
##
##
##bitwiseand=cv2.bitwise_not(frame,frame1)
##cv2.imshow("bitwiseAnd",bitwiseand)










##image=cv2.imread("new.jpg")
##cv2.imshow("original",image)
##
##a=np.ones(image.shape,dtype="uint8")*100
##final=cv2.subtract(image,a)
##cv2.imshow("final",final)
##
####print(str(cv2.add(np.uint8([200]),np.uint8([100]))))
####print(str(cv2.subtract(np.uint8([50]),np.uint8([100]))))












##
##ratio=150.0/image.shape[0]
##dim=(150,int(image.shape[1]*ratio))
##a=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
##re=imutils.resize(image,height=100)
##cv2.imshow("resized",re)

##a=cv2.flip(image,-1)
##cv2.imshow("flip",a)

##crop=image[70:300,200:480]
##cv2.imshow("cropped",crop)





##(h,w)=image.shape[0:2]
##center=(h//2,w//2)
##a=cv2.getRotationMatrix2D(center,90,0.5)
##shifted=cv2.warpAffine(image,a,(w,h))
##cv2.imshow("updated",shifted)




##def shift(image,x,y):
##    
##
##    M=np.float32([[1,0,x],[0,1,y]])
##    shifted=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
##
##    return shifted
##
##updated=shift(image,-90,90)
##cv2.imshow("modified",updated)











##
##canvas=np.zeros((300,300,3),dtype="uint8")
##(centrex,centrey)=(canvas.shape[1]//2, canvas.shape[0]//2)
##white=(255,255,255)
##
##for r in range(0,155,5):
##    cv2.circle(canvas,(centrex,centrey),r,white)
##
##cv2.imshow("cana",canvas)
##
##for i in range(0,25):
##    r=np.random.randint(5,high=200)
##    color=np.random.randint(0,high=256,size=(3,)).tolist()
##    pt=np.random.randint(0,high=300,size=(2,))
##    cv2.circle(canvas,tuple(pt),r,color,-1)
##cv2.imshow("abstract",canvas)
##cv2.waitKey(0)
##
##canvas=np.ones((300,300,3))
##green=(0,255,0)
##cv2.line(canvas,(0,0),(300,300),green)
##
##
##cv2.rectangle(canvas,(10,5),(60,90),green)
##cv2.circle(canvas,(150,150),100,green)
##cv2.imshow("canvas",canvas)
##cv2.waitKey(0)
##
##image=cv2.imread("new.jpg")
##print("width",image.shape[1])
##print("height",image.shape[0])
##print("channels",image.shape[2])

##cv2.imshow("original",image)
##(b,g,r)=image[0,0]
##print("red {} green {} blue {} ".format(r,g,b))
##image[0,0]=(0,0,255)
##(b,g,r)=image[0,0]
##print("red {} green {} blue {} ".format(r,g,b))
##
##left_top_corner=image[0:100,0:100]
##cv2.imshow("left",left_top_corner)
##
##image[0:100,0:100]=(0,255,0)
##
##cv2.imshow("coloured",image)
##cv2.waitKey(0)


