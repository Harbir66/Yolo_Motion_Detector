import cv2
import numpy as np



yolo = cv2.dnn.readNet("./yolov3.weights","./yolov3.cfg")  #Loading pretrained model 

classes = []  # this will store all the classes pre trained model is able to identify 
with open("./coco.names",'r') as f:
    classes=f.read().splitlines()

img = cv2.imread("./demo3.jpg")  #Reading the image
# img=cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4))) #To resize the image

(h,w) = img.shape[:2]          # storing the height and width of image

blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0), swapRB=True, crop=False)   ## here we are resizing image and storing in blob
## here 1/255 is used to normalize the image and ensure that values range from 0 to 1
## (320,320) is to resize image to smaller image as large image would result in more computaion time
## (0,0,0) is tuple of mean values to be substracted from r g b channels of image
## swapRB is to conver image from BGR to RGB channel
## crop is false as we do not crop image

## blob will now have 1 image with 3 channels and dimentions of 320 x 320
# temp = blob[0].reshape(320,320,3)
# cv2.imshow("BLOB",cv2.cvtColor(temp,cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)

yolo.setInput(blob)   # setting input with blob

output_layer_names=yolo.getUnconnectedOutLayersNames()
layeroutput=yolo.forward(output_layer_names)

boxes =[]
confidences=[]
class_ids=[]

for output in layeroutput:
    for detection in output:
        scores=detection[5:]  #detection has first four values as dimentions of the object and remaining values as confidence score of various classes our model is trained on 
        class_id=np.argmax(scores)  # this will index value of class with heighest confidence score
        confidence=scores[class_id] # this will give us confidence score of the said class
        thresh = 0.75   ## thershold this will
        if confidence > thresh :
            center_x=int(detection[0]*w)  # this give center on x axis of box
            center_y=int(detection[1]*h)  # this give cener of y axis of box
            w1=int(detection[2]*w)        # this gives the width of box
            h1=int(detection[3]*h)        # this gives the height of box

            x=int(center_x-w1/2)          # this will give x corr of start of box
            y=int(center_y-h1/2)          # this will give y corr of start of box

            boxes.append([x,y,w1,h1])
            confidences.append(float(confidence))
            class_ids.append(class_id)


total_obects=len(boxes)  # total number of detected objects


indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
font = cv2.FONT_HERSHEY_COMPLEX
colors = np.random.uniform(0, 255, size=(len(boxes), 3)) #list of colors for each possible class label

## This will put boxes along with label and confidence score over detected objects
for i in indexes:
    x,y,w,h = boxes[i]

    label = str(classes[class_ids[i]])
    confi = str(round(confidences[i],2))
    color = colors[i]

    cv2.rectangle(img, (x,y),(x+w,y+h),color, 1)

    text = "{}: {}".format(label, confi)
    cv2.putText(img, text, (x, y - 5), font,0.5, color, 2)
	

print("Total number of detected objects are :",total_obects)
# cv2.imwrite("Img_with_boxes3.jpg",img)       #this is save the image with boxes and labels
cv2.imshow("Image",img)
cv2.waitKey(0)
