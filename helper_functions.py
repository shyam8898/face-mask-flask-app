import base64
import requests
import json
import cv2
def get_b64_string(image):
    image_string=base64.b64encode(image.read())
    image_string=image_string.decode('utf-8')  
    #url = "https://jw5865yno7.execute-api.us-east-1.amazonaws.com/dev/upload"     #sagemaker_dockerized_api
    url = "https://2f57eufe4l.execute-api.us-east-1.amazonaws.com/dev/upload"     #model_dockerized_api
    input_dict = {"input": image_string}
    response = requests.post(url, json=input_dict)
    result = json.loads(response.text)
    final_result=result['image']
    return final_result

def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates

def plot_boxes(results, frame,classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")

    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]


            if text_d == 'mask':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background

                
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

            elif text_d == 'nomask':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0,255), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 0,255), -1) ## for text label background

                
                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
            ## print(row[4], type(row[4]),int(row[4]), len(text_d))

    return frame