from flask import Flask, render_template, request, Response
from helper_functions import get_b64_string, detectx, plot_boxes
import cv2
import torch
import time
  
app = Flask(__name__)

def generate_frames():
    model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True)
    classes = model.names
    global camera
    camera= cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_POS_FRAMES, 3)
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame,1)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results = detectx(frame, model = model)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            frame = plot_boxes(results, frame,classes = classes)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/detectObject',methods=['POST'])
def mask_image():
    try:
        start=time.time()
        image=request.files['image']
        img_base64=get_b64_string(image)
        end=time.time()
        if img_base64!=0:
            return render_template('./result.html',img_base64=img_base64,time=end-start)
        else:
            error_msg="Sorry, you have selected wrong image!"
            return render_template('./error.html',error_message=error_msg)
    except KeyError:
        error_msg="Sorry, you have not selected proper image file!"
        return render_template('./error.html',error_message=error_msg)
    except Exception as e:
        return render_template('./error.html',error_message=e)

@app.route('/') 
def home():  
    return render_template('./index.html')

@app.route('/webcam',methods=['POST']) 
def webcam():  
    return render_template('./webcam_result.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/done')
def done():
    
    if camera.isOpened():
        print("Releasing cam feed")
        camera.release()
    return render_template('./index.html')
  
if __name__ =='__main__':  
    app.run(debug = True)
