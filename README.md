# Yolo_Object_Detector
A simple object detector based on YOLO algorithm 


## **1. Methodology**
![image](https://user-images.githubusercontent.com/62716902/208482711-98b3bdab-2944-4f54-a0ae-8ee6517e5514.png)

## **2. Description**
This poject uses computer vision and yolo algorithm to detect various objects in the given picture and also mark them in the picture by drawing bounding box around the detected object along with the confidence.

## **3. Input**
<br>
<img src=".\demo3.jpg">

## **4. Output**
<br>
<img src=".\Img_with_boxes3.jpg">
<br>

## **5. Live link**
Link: 
https://harbir66-yolo-object-detector-app-oi90li.streamlit.app/

## **6. Screenshots**
![image](https://user-images.githubusercontent.com/62716902/206751283-1bebd135-1905-40b2-a879-855aba19d795.png)


<pre>
TO RUN :-
  Method 1: Online
    Hosted on https://harbir66-yolo-object-detector-app-oi90li.streamlit.app/
    NOTE: Hosted version uses yolo-tiny(less powerfull but small in size)
  
  Method 2: Offline / local machine
    1. Create Virtual Environment on Python using commads :  "pip install virtualenv"  and then : "python -m venv myenv"
    2. Run Virtual Environment by typing  "\myenv\scripts\activate" on comand prompt
    3. Run Command : "pip install -r requirements.txt"
    4. Run Command : "streamlit run app.py"
</pre>

NOTE : You can download weights file from https://pjreddie.com/media/files/yolov3.weights for yolo3 instead of yolo-tiny
