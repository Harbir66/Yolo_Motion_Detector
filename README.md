# Yolo_Object_Detector
A simple object detector based on YOLO algorithm 
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

Sample Input :
<br>
<img src=".\demo3.jpg">

Sample Outputs :
<br>
<img src=".\Img_with_boxes3.jpg">
<br>
ScreenShots of Website:
![image](https://user-images.githubusercontent.com/62716902/206751283-1bebd135-1905-40b2-a879-855aba19d795.png)
