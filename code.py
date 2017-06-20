import numpy as np
import cv2
import math
import datetime


############ INPUT VIDEO ###########
cap=cv2.VideoCapture('sample.wmv')

#  VIDEO PROPERTIES
print "Frame Width : "
print cap.get(3)  #Frame Width
print "Frame Height :"
print cap.get(4)  #Frame Height
#print cap.grab()
fps = cap.get(cv2.CAP_PROP_FPS)
print "FPS :",fps


# Tracked list contains ID's of tracked objects
trackedlist=set()


# Center positions array is used to store center postions of tracked objects
centerPositions=[]
blobarea=[]
aspectrat=[]
timeblob=[]
intime= [0]*100000
outtime= [0]*100000
global blobs
blobs=[]


# Define kernel for performing morpholigical operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))


ret,imgFrame1Copy=cap.read()
ret,imgFrame2Copy=cap.read()
carCount=0
carCount2 = 0
carCount3 = 0
carCount4 = 0
front=0   # For counting cars entering from front side
back=0    # For counting cars entering from back side
vehicle=0
blnFirstFrame = True
fps=0

print "Enter initial X for LINE1 :"
x1_s=input()
print "Enter initial y for LINE1 :"
y1_s=input()
print "Enter final X for LINE1 :"
x1_e=input()
print "Enter final Y for LINE1 :"
y1_e=input()

print "Enter initial X for LINE2 :"
x2_s=input()
print "Enter initial y for LINE2 :"
y2_s=input()
print "Enter final X for LINE2 :"
x2_e=input()
print "Enter final Y for LINE2 :"
y2_e=input()

     ###### LINE 1 ######(2-D array storing starting and end points of line)
Line1=np.zeros((2,2),np.float32)
horizontalLine1=((imgFrame2Copy.shape[0])*0.70)
horizontalLine3=((imgFrame2Copy.shape[1])*0.30)
#Line1[0][0]= horizontalLine3
#Line1[0][1]= horizontalLine1
#Line1[1][0]= imgFrame2Copy.shape[1] - 1
#Line1[1][1] =horizontalLine1

Line1[0][0]= ((imgFrame2Copy.shape[1])*x1_s)
Line1[0][1]= ((imgFrame2Copy.shape[0])*y1_s)
Line1[1][0]= ((imgFrame2Copy.shape[1])*x1_e)
Line1[1][1]= ((imgFrame2Copy.shape[0])*y1_e)



     ###### LINE 2 ######(2-D array storing starting and end points of line)
Line2=np.zeros((2,2),np.float32)
horizontalLine2=((imgFrame2Copy.shape[0])*0.85)
horizontalLine3=((imgFrame2Copy.shape[1])*0.30)
#Line2[0][0]= horizontalLine3
#Line2[0][1]= horizontalLine2
#Line2[1][0]= imgFrame2Copy.shape[1] - 1
#Line2[1][1] =horizontalLine2

Line2[0][0]= ((imgFrame2Copy.shape[1])*x2_s)
Line2[0][1]= ((imgFrame2Copy.shape[0])*y2_s)
Line2[1][0]= ((imgFrame2Copy.shape[1])*x2_e)
Line2[1][1]= ((imgFrame2Copy.shape[0])*y2_e)

# Print Line1 and Line 2
print "Line1: "
print Line1
print "Line2: "
print Line2


def millis_interval(start,end):
    """start and end are datetime instances"""
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis


#define Blob Filter for blob analysis and Filtering of Bad Blobs

class blobz(object): 
    def __init__(self,contour):
        global currentContour 
        global currentBoundingRect 
        global centerPosition
        global centerPositions
        global timeblob
        global currentime
        global cx
        global cy
        global dblCurrentDiagonalSize 
        global dblCurrentAspectRatio 
        global intCurrentRectArea
        global blnCurrentMatchFoundOrNewBlob 
        global blnStillBeingTracked 
        global intNumOfConsecutiveFramesWithoutAMatch 
        global predictedNextPosition
        global numPositions
	global blobarea
	global aspectrat
        self.predictedNextPosition=[]
        self.centerPosition=[]
	self.blobarea=[]
        currentBoundingRect=[]
        currentContour=[]
        self.centerPositions=[]
        self.currentContour=contour
	self.aspectrat=[]
	self.timeblob=[]
        self.currentBoundingArea=cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        self.currentBoundingRect=[x,y,w,h]
        cx=(2*x+w)/2
        cy=(2*y+h)/2
        self.centerPosition=[cx,cy]
	self.currentime=datetime.datetime.now()
        self.dblCurrentDiagonalSize=math.sqrt(w*w+h*h)
        self.dblCurrentAspectRatio=(w/(h*1.0))
        self.intCurrentRectArea=w*h
	self.blobarea.append(self.intCurrentRectArea)
        self.blnStillBeingTracked = True
        self.blnCurrentMatchFoundOrNewBlob = True
        self.intNumOfConsecutiveFramesWithoutAMatch = 0
        self.centerPositions.append(self.centerPosition)    
	self.aspectrat.append(self.dblCurrentAspectRatio)
	self.timeblob.append(self.currentime)
 
 
 #Next Position Prediction Algorithm based on last 5 weighing sum of tracked blob positions
    def predictNextPosition(self):
        numPositions=len(self.centerPositions)
        if(numPositions == 1):
            self.predictedNextPosition=[self.centerPositions[-1][-2],self.centerPositions[-1][-1]]
        if(numPositions == 2):
            deltaX = self.centerPositions[1][0]-self.centerPositions[0][0]
            deltaY =self.centerPositions[1][1] -self.centerPositions[0][1]
            self.predictedNextPosition =[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]
        if(numPositions == 3):
            sumOfXChanges= ((self.centerPositions[2][0] - self.centerPositions[1][0]) * 2) +((self.centerPositions[1][0] - self.centerPositions[0][0]) * 1)
            deltaX=(sumOfXChanges / 3)
            sumOfYChanges= ((self.centerPositions[2][1] - self.centerPositions[1][1]) * 2) +((self.centerPositions[1][1] - self.centerPositions[0][1]) * 1)
            deltaY=(sumOfYChanges / 3)
            self.predictedNextPosition=[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]
        if(numPositions == 4):
            sumOfXChanges= ((self.centerPositions[3][0] - self.centerPositions[2][0]) * 3) +((self.centerPositions[2][0] - self.centerPositions[1][0]) * 2) +((self.centerPositions[1][0] - self.centerPositions[0][0]) * 1)
            deltaX=(sumOfXChanges / 6)
            sumOfYChanges= ((self.centerPositions[3][1] - self.centerPositions[2][1]) * 3) +((self.centerPositions[2][1] - self.centerPositions[1][1]) * 2) +((self.centerPositions[1][1] - self.centerPositions[0][1]) * 1)
            deltaY= (sumOfYChanges / 6)
            self.predictedNextPosition=[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]
        if (numPositions >= 5):
            sumOfXChanges= ((self.centerPositions[numPositions-1][0] - self.centerPositions[numPositions-2][0]) * 4) +((self.centerPositions[numPositions-2][0] - self.centerPositions[numPositions-3][0]) * 3) +((self.centerPositions[numPositions-3][0] - self.centerPositions[numPositions-4][0]) * 2) +((self.centerPositions[numPositions-4][0] - self.centerPositions[numPositions-5][0]) * 1)
            sumOfYChanges= ((self.centerPositions[numPositions-1][1] - self.centerPositions[numPositions-2][1]) * 4) +((self.centerPositions[numPositions-2][1] - self.centerPositions[numPositions-3][1]) * 3) +((self.centerPositions[numPositions-3][1] - self.centerPositions[numPositions-4][1]) * 2) +((self.centerPositions[numPositions-4][1] - self.centerPositions[numPositions-5][1]) * 1)
            deltaX= (sumOfXChanges / 10)
            deltaY=(sumOfYChanges / 10)
            self.predictedNextPosition=[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]


# Funtion to check if the currentframeblobs match to existing blobs
def matchCurrentFrameBlobsToExistingBlobs(blobs,currentFrameBlobs):
    i=0
    for existingBlob in blobs:
        existingBlob.blnCurrentMatchFoundOrNewBlob = False
        existingBlob.predictNextPosition()
    for currentFrameBlob in currentFrameBlobs:
        intIndexOfLeastDistance = 0
        dblLeastDistance = 1000000.0
        for i in range(len(blobs)):
            if(blobs[i].blnStillBeingTracked == True):
                dblDistance=distanceBetweenPoints(currentFrameBlob.centerPositions[-1],blobs[i].predictedNextPosition)
                if(dblDistance < dblLeastDistance):
                    dblLeastDistance = dblDistance
                    intIndexOfLeastDistance = i
        if(dblLeastDistance < (currentFrameBlob.dblCurrentDiagonalSize * 1.0)/1.2):
            blobs=addBlobToExistingBlobs(currentFrameBlob, blobs, intIndexOfLeastDistance)
	    x=blobs[i].centerPositions[-1][-2]
	    y=blobs[i].centerPositions[-1][-1]
	    if (x>=Line1[0][0] and y>=Line1[0][1] and y<=Line2[0][1]):
		    print "Tracking ID:",i
		    trackedlist.add(i)
	    if len(blobs[i].centerPositions)>=2:
		    prevFrameIndex= len(blobs[i].centerPositions) - 2
            	    currFrameIndex= len(blobs[i].centerPositions) - 1
	    	    if (blobs[i].centerPositions[prevFrameIndex][-1] > Line2[0][1] and blobs[i].centerPositions[currFrameIndex][-1] <= Line2[0][1] and blobs[i].centerPositions[prevFrameIndex][-2] >=Line2[0][0] and blobs[i].centerPositions[currFrameIndex][-2]>=Line2[0][0]):
			    #print "Entered"
			    intime[i]=datetime.datetime.now()
	            if (blobs[i].centerPositions[prevFrameIndex][-1] > Line1[0][1] and blobs[i].centerPositions[currFrameIndex][-1] <= Line1[0][1] and blobs[i].centerPositions[prevFrameIndex][-2] >=Line2[0][0] and blobs[i].centerPositions[currFrameIndex][-2]>=Line2[0][0]):
			    #print "Exit"
			    outtime[i]=datetime.datetime.now()
	    	    if (blobs[i].centerPositions[prevFrameIndex][-1] < Line1[0][1] and blobs[i].centerPositions[currFrameIndex][-1] >= Line1[0][1] and blobs[i].centerPositions[prevFrameIndex][-2] >=Line2[0][0] and blobs[i].centerPositions[currFrameIndex][-2]>=Line2[0][0]):
			    #print "Entered"
			    intime[i]=datetime.datetime.now()
	    	    if (blobs[i].centerPositions[prevFrameIndex][-1] < Line2[0][1] and blobs[i].centerPositions[currFrameIndex][-1] >= Line2[0][1] and blobs[i].centerPositions[prevFrameIndex][-2] >=Line2[0][0] and blobs[i].centerPositions[currFrameIndex][-2]>=Line2[0][0]):
			    #print "Exit"
			    outtime[i]=datetime.datetime.now()
		    #if (blobs[i].centerPositions[currFrameIndex][-1] <= Line2[0][1] and  blobs[i].centerPositions[currFrameIndex][-1]>=Line1[0][1]):
			    #	    timeblob[i].append(datetime.datetime.now())
        else:
            blobs,currentFrameBlob=addNewBlob(currentFrameBlob, blobs)
    for existingBlob in blobs:
        if(existingBlob.blnCurrentMatchFoundOrNewBlob == False):
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch = existingBlob.intNumOfConsecutiveFramesWithoutAMatch + 1
        if(existingBlob.intNumOfConsecutiveFramesWithoutAMatch >=3):
            existingBlob.blnStillBeingTracked =False
    return blobs   


#Function to find distance between two points
def distanceBetweenPoints(pos1,pos2):
    if (pos2==[]):
        dblDistance=math.sqrt((pos1[0])**2+(pos1[1])**2)
    else:
        dblDistance=math.sqrt((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2)
    return dblDistance


# Function to add currentframeblob to existing blob
def addBlobToExistingBlobs(currentFrameBlob, blobs, intIndex):
    blobs[intIndex].currentContour = currentFrameBlob.currentContour
    blobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect
    blobs[intIndex].blobarea.append(currentFrameBlob.intCurrentRectArea)
    blobs[intIndex].centerPositions.append(currentFrameBlob.centerPositions[-1])
    blobs[intIndex].aspectrat.append(currentFrameBlob.dblCurrentAspectRatio)
    blobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize
    blobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio
    blobs[intIndex].timeblob.append(currentFrameBlob.currentime)
    blobs[intIndex].blnStillBeingTracked = True
    blobs[intIndex].blnCurrentMatchFoundOrNewBlob = True
    return blobs


# Funtion to add new blobs
def addNewBlob(currentFrameBlob,Blobs):
    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = True
    blobs.append(currentFrameBlob)
    return blobs,currentFrameBlob


#Draw Blob Information on Image#
def drawBlobInfoOnImage(blobs,m1):
    for i in range(len(blobs)):
        if (blobs[i].blnStillBeingTracked == True):
            x,y,w,h=blobs[i].currentBoundingRect
	    if(x>=Line1[0][0]):
           	 cv2.rectangle(m1,(x, y), (x + w, y + h), (255, 0, 0), 2)
    return m1                             


#Draw Car Count On Image#
def drawCarCountOnImage(carCount,carCount2,carCount3,carCount4,front,back,m1,fps):
    front=(min(carCount,carCount2))
    carCount=front
    carCount2=front
    back=(min(carCount3,carCount4))
    carCount3=back
    carCount4=back
    initText = "Front: "
    text =initText+str(front) +" Back : " + str(back) 
    cv2.putText(m1, "             {} ".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return m1


# Function to check if blob crossed the line2
def checkIfBlobsCrossedTheLine(blobs,Line2,carCount,carCount4):
    atLeastOneBlobCrossedTheLine= False
    for blob in blobs:
        if (blob.blnStillBeingTracked == True and len(blob.centerPositions) >= 2):
            prevFrameIndex= len(blob.centerPositions) - 2
            currFrameIndex= len(blob.centerPositions) - 1
	    if (blob.centerPositions[prevFrameIndex][-1] > Line2[0][1] and blob.centerPositions[currFrameIndex][-1] <= Line2[0][1] and blob.centerPositions[prevFrameIndex][-2] >=Line2[0][0] and blob.centerPositions[currFrameIndex][-2]>=Line2[0][0]):
	        carCount += 1
                atLeastOneBlobCrossedTheLine = True
	    if (blob.centerPositions[prevFrameIndex][-1] <Line2[0][1] and blob.centerPositions[currFrameIndex][-1] >= Line2[0][1] and blob.centerPositions[prevFrameIndex][-2] >=Line2[0][0] and blob.centerPositions[currFrameIndex][-2]>=Line2[0][0]):
		    #print "YEs"
                    carCount4 += 1
    return atLeastOneBlobCrossedTheLine,carCount,carCount4


# Function to check if blob crossed the line1
def checkIfBlobs2CrossedTheLine(blobs,Line1,carCount2,carCount3):
    atLeastOneBlobCrossedTheLine2= False
    for blob in blobs:
        if (blob.blnStillBeingTracked == True and len(blob.centerPositions) >= 3):
            prevFrameIndex= len(blob.centerPositions) - 2
            #print len(blob.centerPositions)
            #print prevFrameIndex
            currFrameIndex= len(blob.centerPositions) - 1
	    if (blob.centerPositions[prevFrameIndex][-1] > Line1[0][1] and blob.centerPositions[currFrameIndex][-1] <= Line1[0][1] and blob.centerPositions[prevFrameIndex][-2] >=Line2[0][0] and blob.centerPositions[currFrameIndex][-2]>=Line2[0][0] and blob.centerPositions[prevFrameIndex-1][-1]>Line1[0][1]):
	        carCount2 += 1
                atLeastOneBlobCrossedTheLine2 = True
            if (blob.centerPositions[prevFrameIndex][-1] < Line1[0][1] and blob.centerPositions[currFrameIndex][-1] >= Line1[0][1] and blob.centerPositions[prevFrameIndex][-2] >=Line2[0][0] and blob.centerPositions[currFrameIndex][-2]>=Line2[0][0] and blob.centerPositions[prevFrameIndex-1][-1]<Line1[0][1] ):
		    #print "NO"
		    carCount3 +=  1
		    #print carCount3
    return atLeastOneBlobCrossedTheLine2,carCount2,carCount3



##### MAIN PROGRAM BEGINS ######

x=0 #Blob count
y=0 #Blob Verification
while(True):
#print "Start"
    startTime=datetime.datetime.now()
    #print " Start time:" , startTime
    m1=imgFrame1Copy
    n1=imgFrame2Copy
    a1 = cv2.cvtColor(imgFrame1Copy,cv2.COLOR_BGR2GRAY)
    b1 = cv2.cvtColor(imgFrame2Copy,cv2.COLOR_BGR2GRAY)
    a2 = cv2.GaussianBlur(a1,(5,5),0)
    b2 = cv2.GaussianBlur(b1,(5,5),0)
    imgDifference=cv2.absdiff(b2,a2)
   # cv2.imshow('diff',imgDifference)
    ret1,th1 = cv2.threshold(imgDifference,30,255,cv2.THRESH_BINARY)
    th1 = cv2.dilate(th1,kernel,iterations = 1)
    th1 = cv2.dilate(th1,kernel,iterations = 1)
    fgmask = cv2.erode(th1,kernel,iterations = 1)
    #cv2.imshow('diff',fgmask)
    frameNo=cap.get(1)
    fgmask = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2)
    fg2=np.zeros((fgmask.shape[0],fgmask.shape[1],3),np.uint8)
    fg3=np.zeros((fgmask.shape[0],fgmask.shape[1],3),np.uint8)
    cv2.waitKey(20)
    _,contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fg2, contours, -1, (255,255,255), -1)
    #cv2.imshow('contour',fg2)
    hulls=[]
    for c in range(len(contours)):
        hull=cv2.convexHull(contours[c])
        hulls.append(hull)
    curFrameblobs=[]
    for c in range(len(hulls)):
        ec=blobz(hulls[c])
	### INITIALLY FILTER BAD BLOBS ###
        if(ec.intCurrentRectArea>100 and ec.dblCurrentAspectRatio>=0.2 and ec.dblCurrentAspectRatio<=1.2 and ec.dblCurrentDiagonalSize>30 and ec.currentBoundingRect[2]>20 and ec.currentBoundingRect[3]>20 and (ec.currentBoundingArea*1.0/ec.intCurrentRectArea)>.4):
            curFrameblobs.append(ec)
    if (blnFirstFrame ==True):
        for f1 in curFrameblobs:
            blobs.append(f1)
    else: 
        blobs=matchCurrentFrameBlobsToExistingBlobs(blobs,curFrameblobs)                     
    m1=drawBlobInfoOnImage(blobs,m1)
    fg2=drawBlobInfoOnImage(blobs,fg2)
    #cv2.imshow('finale',fg2)
    blob2 = blobs
    #cv2.imshow('original1',m1)
    atLeastOneBlobCrossedTheLine,carCount,carCount4=checkIfBlobsCrossedTheLine(blobs, Line2, carCount,carCount4)
    atLeastOneBlobCrossedTheLine2,carCount2,carCount3=checkIfBlobs2CrossedTheLine(blobs, Line1, carCount2,carCount3)

    if (atLeastOneBlobCrossedTheLine):
        cv2.line(m1,(Line2[0][0],Line2[0][1]),(Line2[1][0],Line2[1][1]),(0,255,0), 2)
        x=x+1
	#print x
        
    if (atLeastOneBlobCrossedTheLine == 0):
        cv2.line(m1,(Line2[0][0],Line2[0][1]),(Line2[1][0],Line2[1][1]),(0,0,255), 2)
    
    if (atLeastOneBlobCrossedTheLine2):
        cv2.line(m1,(Line1[0][0],Line1[0][1]),(Line1[1][0],Line1[1][1]),(255,255,255), 2)
        y=y+1
        #print x
        
    if (atLeastOneBlobCrossedTheLine2 ==0):
        cv2.line(m1,(Line1[0][0],Line1[0][1]),(Line1[1][0],Line1[1][1]),(0,0,0), 2)

    
    m1=drawCarCountOnImage(carCount,carCount2,carCount3,carCount4,front,back,m1,fps)
    endTime=datetime.datetime.now()
    millis=millis_interval(startTime,endTime)
    fps=(1.0*1000)/millis
    cv2.imshow('info',m1)
    cv2.waitKey(1)
    imgFrame1Copy=imgFrame2Copy
    ret,imgFrame2Copy=cap.read()
    if not ret:
        break
    blnFirstFrame = False

    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
print "\n\n\n"
print "Tracked ID set : "
print trackedlist
#print "Center Positions array:"
i=0
n= len(trackedlist)
file =open("intimendouttime.txt","w+")
file.write("\n\n\n INTIME AND OUTTIMES \n\n\n")
#print"\n\n\n INTIME AND OUTTIME\n"


#### PRINT INTIME AND OUTTIME AND WRITE TO A FILE ####
while (i<n) :
#	print "intime[",list(trackedlist)[i],"]:",intime[list(trackedlist)[i]]
#	print "outtime[",list(trackedlist)[i],"]:",outtime[list(trackedlist)[i]]
	file.write("intime["+repr(list(trackedlist)[i])+ "]:" + repr(intime[list(trackedlist)[i]]) +"\n")
	file.write("outtime["+repr(list(trackedlist)[i])+ "]:" + repr(outtime[list(trackedlist)[i]]) +"\n\n\n")
	i+=1
file.close()


#file =open("time.txt","w+")
#file.write("\n\n\n TIME \n\n\n")
#i=0
#while (i<n) :
#	file.write("time["+repr(list(trackedlist)[i])+ "]:" + repr(time[list(trackedlist)[i]]) +"\n")
#	i+=1
#file.close()

i=0
xy=0
yz=0
zx=0
xyz=0
#### PRINT CENTER POSTIONS AND WRITE TO A FILE #####
#file= open("centeroutput.txt","w+")
#file.write("\n\n\n CenterPositions :\n\n\n")
#while (i<n):
#	print "CenterPosition (" ,list(trackedlist)[i],"):\n",blobs[list(trackedlist)[i]].centerPositions
#	file.write("CenterPosition (" + repr(list(trackedlist)[i]) + "):\n" + repr(blobs[list(trackedlist)[i]].centerPositions) +"\n\n" )
#	i+=1
#file.close()

#### PRINT BLOB AREA AND WRITE TO A FILE ####
#file= open("areaout.txt","w+")
#file.write("\n\n\n BLOB AREA :\n\n\n")
#i=0
#while (i<n):
#print "blobarea (" ,list(trackedlist)[i],"):\n",blobs[list(trackedlist)[i]].blobarea
#	file.write("Blob Area (" + repr(list(trackedlist)[i]) + "):\n" + repr(blobs[list(trackedlist)[i]].blobarea) +"\n\n" )
#	i+=1
#file.close()


#file= open("time.txt","w+")
#file.write("\n\n\n TIME :\n\n\n")
#i=0
#while (i<n):
#print "blobarea (" ,list(trackedlist)[i],"):\n",blobs[list(trackedlist)[i]].blobarea
#	file.write("TIME (" + repr(list(trackedlist)[i]) + "):\n" + repr(blobs[list(trackedlist)[i]].timeblob) +"\n\n" )
#	i+=1
#file.close()

#### PRINT ASPECT RATIO AND WRITE TO A FILE ####
#file= open("aspectratio.txt","w+")
#file.write("\n\n\n BLOB ASPECT RATIO :\n\n\n")
#i=0
#while (i<n):
#	print "blobaspect (" ,list(trackedlist)[i],"):\n",blobs[list(trackedlist)[i]].aspectrat
#	file.write("blobaspectratio (" + repr(list(trackedlist)[i]) + "):\n" + repr(blobs[list(trackedlist)[i]].aspectrat) +"\n\n" )
#	i+=1
#file.close()


####### TO WRITE ALL INFO OF BLOBS ######

file= open("info.txt","w+")
file.write("\n\n\n All things :\n\n\n")
i=0
while (i<n):
	j=0
	#for j in blobs[list(trackedlist)[i]].centerPositions:
	#	print "%r " % j
	file.write( " \n\n\n\n For tracking ID : " + repr(list(trackedlist)[i])+ "\n")
	while j< len(blobs[list(trackedlist)[i]].centerPositions):
		#print blobs[list(trackedlist)[i]].centerPositions[j] , " ", blobs[list(trackedlist)[i]].aspectrat[j] ,"\n"
		file.write("time :" + repr(blobs[list(trackedlist)[i]].timeblob[j]) +" " + "Centrepositions :" + repr(blobs[list(trackedlist)[i]].centerPositions[j]) +" "  + "Areablob :" + repr(blobs[list(trackedlist)[i]].blobarea[j]) +" " "blobaspectratio :" + repr(blobs[list(trackedlist)[i]].aspectrat[j]) +"\n")
		j+=1
	i+=1
file.close()
