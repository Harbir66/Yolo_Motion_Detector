import streamlit as st
import cv2
import numpy as np


from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide", page_title="Object Detection", page_icon=":eyes:")
# st.markdown(""" <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# header {visibility: hidden;}
# </style> """, unsafe_allow_html=True)

yolo = cv2.dnn.readNet("./yolov3-tiny.weights","./yolov3-tiny.cfg")
# yolo = cv2.dnn.readNet("./yolov3.weights","./yolov3.cfg") 

classes = []  
with open("./coco.names",'r') as f:
    classes=f.read().splitlines()



st.write("## Detect Different Objects in an Image :eyes:")
st.write(
    ":dog: Try uploading an image to watch the Objects magically Detected. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/Harbir66/Yolo_Object_Detector) on GitHub. Special thanks to the [yolo](https://pjreddie.com/darknet/yolo/) :grin:"
)
st.sidebar.write("## Upload and download :gear: ")
thresh=st.sidebar.select_slider("Threshold",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],value=0.5)
# total_obects=0
boxes =[]
confidences=[]
class_ids=[]


# Download the fixed image
def convert_image(image):
    img=Image.fromarray(image)
    buf = BytesIO()
    img.save(buf, format="png")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    image1=np.array(image)

    #yolo
    (h,w) = image1.shape[:2]  

    blob = cv2.dnn.blobFromImage(image1, 1/255, (640,640), (0,0,0), swapRB=True, crop=False)

    yolo.setInput(blob)

    output_layer_names=yolo.getUnconnectedOutLayersNames()
    layeroutput=yolo.forward(output_layer_names)    
    


    for output in layeroutput:
        for detection in output:
            scores=detection[5:]  
            class_id=np.argmax(scores)  
            confidence=scores[class_id] 
            
            if confidence > thresh :
                center_x=int(detection[0]*w)  
                center_y=int(detection[1]*h)  
                w1=int(detection[2]*w)        
                h1=int(detection[3]*h)        

                x=int(center_x-w1/2)          
                y=int(center_y-h1/2)          

                boxes.append([x,y,w1,h1])
                confidences.append(float(confidence))
                class_ids.append(class_id)

  

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    fixed=image.copy()
    fixed=np.array(fixed)


    total_obects = 0
    for i in indexes:
        total_obects+=1
        x,y,w,h = boxes[i]

        label = str(classes[class_ids[i]])
        confi = str(round(confidences[i],2))
        color = colors[i]

        cv2.rectangle(fixed, (x,y),(x+w,y+h),color, 1)

        text = "{}: {}".format(label, confi)
        cv2.putText(fixed, text, (x, y - 5), font,1, color, 2)


    st.write("## Results :eyes:")
    st.write("### Total number of detected objects are :",total_obects," :eyes:")
    for i in indexes:
        st.write(" ",str(classes[class_ids[i]])," : ",str(round(confidences[i],2)) )
    #yolo end
    col1.write("Original Image :camera:")
    col1.image(image)

    col2.write("Detections :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])



if my_upload is not None:
    fix_image(upload=my_upload)
else:
    fix_image(".\demo1.jpg")

st.sidebar.write("## Made with :heart: by Harbir Singh")
st.sidebar.write("## 101917050")
st.sidebar.write("## CSE2")