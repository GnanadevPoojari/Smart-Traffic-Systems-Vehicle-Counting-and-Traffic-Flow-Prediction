import threading
import streamlit as st
import cv2
import csv
import collections
import numpy as np
# from google.colab.patches import cv2_imshow
from datetime import datetime,timedelta
import time
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Your existing imports and functions...
# Vehicle-Tracker

import math
st.set_page_config(layout="wide")
# [theme]
# backgroundColor = "#F0F0F0"
#st.set_page_config(layout="wide", backgroundColor="#f0f0f0")
st.markdown("""
<style>
body {
  background: white; 
  background: -webkit-linear-gradient(to right, #ff0099, #493240); 
  background: linear-gradient(to right, #ff0099, #493240); 
}
</style>
    """, unsafe_allow_html=True)



class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h, index = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id, index])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, index = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids



def ad(a, b):
    return a+b


# Initialize Tracker
tracker = EuclideanDistTracker()


# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 255,0)
font_size = 0.5
font_thickness = 2

vehicle_counts = []
timestamps = []
interval_seconds = 1
append_interval = 1
start_time = time.time()
# Middle cross line position
middle_line_position = 225
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15


# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')


# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

## Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)



# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

video_date = datetime(2024, 3, 28,8,0,0)

def write_to_csv():
    global vehicle_counts, timestamps
    global up_list,down_list
    global video_date
    with open("data2.csv", 'a') as f1:
        # cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        cwriter = csv.writer(f1)
        #to show the real time
        #datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #to fix the date and time

        total_vehicles = sum(up_list) + sum(down_list)
        cwriter.writerow([video_date,*up_list,*down_list,total_vehicles])
        video_date = video_date + timedelta(seconds=1)
        cwriter = csv.writer(f1)

    print("Data saved at 'data2.csv'")

    f1.close()

# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy

# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# Function for count vehicle
def count_vehicle(box_id, img):
    global start_time
    global interval_frames
    global frame_counter
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)

    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1


    if frame_counter >= interval_frames:
        print("hi")
        vehicle_counts.append(up_list + down_list)
        timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    #print(indices)
    for i in indices:
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score
        cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)

displayCount = []
def realTime(video_path):
    # Initialize the videocapture object
    # print(video_path)
    global displayCount

    cap = cv2.VideoCapture(video_path.name)
    # with open("data2.csv", 'w') as f1:
    #         cwriter = csv.writer(f1)
    #         cwriter.writerow(['Timestamp', 'car_up', 'motorbike_up', 'bus_up', 'truck_up', 'car_down', 'motorbike_down', 'bus_down', 'truck_down','Total_Vehicles'])
    input_size = 320
    global interval_frames
    interval_frames = 1 * cap.get(cv2.CAP_PROP_FPS)
    global frame_counter
    frame_counter=0


    global vehicle_counts, timestamps
    # global frame_counter
   # print("interval_frames",interval_frames)  # Calculate frames for 5-second interval
    count = 0
    while True:
        success, img = cap.read()
        frame_counter += 1
        if frame_counter >= interval_frames:
            write_to_csv()
            count+=1
            frame_counter = 0
            print("Data saved for interval of 5 seconds")

        img = cv2.resize(img,(0,0),None,0.5,0.5)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()


        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
        # Feed data to the network
        outputs = net.forward(outputNames)

        # Find the objects from the network output
        postProcess(outputs,img)

        # Draw the crossing lines

        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

        # Draw counting texts in the frame
        cv2.putText(img, "Total vehicles:"+ " "+str(sum(up_list)+sum(down_list)), (420, 120), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Up", (510, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down", (550, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (420, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (420, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (420, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (420, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # Show the frames
        cv2.imshow('image',img)
        # st.write("Counting started...")
        if cv2.waitKey(1) == ord('q'):
            break
        if(count>4):
            break
    displayCount= up_list+down_list
    

    cap.release()
    cv2.destroyAllWindows()
    return displayCount


def prediction(csv_path):
  data = pd.read_csv(csv_path.name)
  # Step 2: Preprocess the data
  # Parse timestamps to extract day and time duration
  data['Timestamp'] = pd.to_datetime(data['Timestamp'])
  data['Day'] = data['Timestamp'].dt.day_name() + "\n(Total_Vehicles)"
  data['Time'] = data['Timestamp'].dt.hour
  data['Total_Vehicles'] = data['Total_Vehicles']
  
  # Aggregate total vehicles by day and hour
  traffic_data = data.groupby(['Day', 'Time'])['Total_Vehicles'].max().reset_index()
  # print(traffic_data)
  # Step 3: Prepare the data for time series forecasting
  # Create a pivot table to make time series data
  traffic_pivot = traffic_data.pivot(index='Time', columns='Day', values='Total_Vehicles').fillna(0)
  print(traffic_pivot)
  
  # Step 4: Train the time series forecasting model
  # Choose a specific day and time to forecast (e.g., Monday at 7 AM)
  forecast_day = 'Monday'
  forecast_time = 7
  
  # Extract the historical data for the chosen day and time
  historical_data = traffic_pivot.loc[forecast_time].values
  print(historical_data)
  # Train an ARIMA model
  model = ARIMA(historical_data)  # Example order, you may need to tune this
  model_fit = model.fit()
  
  # Step 5: Make predictions for the chosen day and time
  # Predict the traffic for the next time slot (e.g., Monday at 8 AM)
  forecast_value = model_fit.forecast(steps=1)[0]
  #print(forecast_value)
  # Step 6: Determine if the traffic is high or not based on the threshold
  threshold = 5
  predicted_traffic = 'High' if forecast_value >= threshold else 'Smooth'
  
  # Step 7: Output the prediction
  print(f"Predicted Traffic on {forecast_day} at {forecast_time}:00: {predicted_traffic}")
  return (traffic_pivot,f"Predicted Traffic on {forecast_day} at {forecast_time}:00: {predicted_traffic}")
  
st.title("Vehicle Counting and Classification")
col1,col2= st.columns([1,2])
with col1:
    video_path = st.file_uploader("Upload a video file", type=["mp4"])
with col2:
    csv_path = st.file_uploader("Upload a csv file", type=["csv"])

if video_path:
    # print(video_path)
    # output= prediction()
    # print(output)
    with col1:
        st.write(f"Video loaded: {video_path.name}")
        start_btn = st.button("Count Vehicles")
        
    
    if start_btn:
        with col1:
            t=st
            t.write("Counting started...")
    
            
        output = realTime(video_path)
        t.empty()
        st.write("Completed Successfully")
        with col1:
            
            st.write("cars : "+str(output[0]+output[4]))
            st.write("MotorBike : "+str(output[1]+output[5]))            
            st.write("Bus : "+str(output[2]+output[6]))          
            st.write("Truck : "+str(output[3]+output[7]))           
            st.write("Total Vehicles : "+str(sum(output)))

if csv_path:
    with col2:
        st.write(f"CSV loaded: {csv_path.name}")
        # st.write(output)
        start_btn = st.button("Predict traffic")
    # stop_btn = st.button("Stop Counting")
    stop=False
    
    if start_btn:
        # st.write("Counting started...")
        output = prediction(csv_path)
        with col2:

            st.write(output[0])
            st.write(output[1])


        

    # if stop_btn:
    #     stop=True
    #     realTime(video_path,stop)
    #     st.write("Counting stopped.")
    
