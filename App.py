import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import os
import requests
from urllib.parse import unquote
import pyrebase
from IPython import display
import ultralytics
import supervision as sv
from ultralytics import YOLO
import collections
import datetime as dt


app = Flask(__name__)


def extract_filename_from_url(url):
    start_index = url.find('%2F') + 3  # Mengabaikan karakter '%2F'
    end_index = url.find('.mp4') + 4   # Menambahkan 4 karakter untuk '.mp4'

    extracted_text = url[start_index:end_index]

    return unquote(extracted_text)


@app.route('/')
def home():
    return 'Hello World'


@app.route('/receive_data', methods=['GET','POST'])
def receive_data():
    try:
        api_url = "https://tp.kenaja.id/v1/api/records?unanalyzed=1"  # Change the endpoint
        response = requests.get(api_url)

        if response.status_code == 200:
            data_from_api = response.json()

            records = data_from_api['data']['records']

            if records:
                record = records[0]  # Get the first record from the response
                record_id = record['id']
                video_link = extract_filename_from_url(record['media']['url'])
                record_data = {'record_id': record_id, 'video_link': video_link}

                return jsonify(record_data), 200
            else:
                return jsonify({'error': 'No records available.'}), 404
        else:
            return jsonify({'error': 'Failed to fetch data from the API.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict',methods = ['POST','GET'])
async def predict():
        HOME = os.getcwd()

        # Mengambil data dari receive_data endpoint
        receive_data_response = requests.get(url_for('receive_data', _external=True))
        if receive_data_response.status_code == 200:
            record_data = receive_data_response.json()

            firebaseConfig = {
                "apiKey": "AIzaSyAXkEQ11G_jDlMd1WHH6B58hu1UD9ohJv0",
                "authDomain": "traffic-pulse-app.firebaseapp.com",
                "projectId": "traffic-pulse-app",
                "storageBucket": "traffic-pulse-app.appspot.com",
                "messagingSenderId": "518077601368",
                "appId": "1:518077601368:android:c28e9be7621fb806095c2d",
                "databaseURL": "https://traffic-pulse-app.firebaseio.com",

            }

            firebase = pyrebase.initialize_app(firebaseConfig)
            storage = firebase.storage()

            path_on_cloud = "captures/"+str(record_data['video_link'])

            storage.child(path_on_cloud).download(path=HOME,filename="test.mp4")

        SOURCE_VIDEO_PATH = f"{HOME}/test.mp4"

        display.clear_output()

        ultralytics.checks()

        display.clear_output()

        print("supervision.__version__:", sv.__version__)

        MODEL = "yolov8x.pt"

        model = YOLO(MODEL)
        model.fuse()

        # dict maping class_id to class_name
        CLASS_NAMES_DICT = model.model.names

        # class_ids of interest - car, motorcycle, bus and truck
        selected_classes = [2, 3, 5, 7]

        LINE_START = sv.Point(50, 1500)
        LINE_END = sv.Point(3840-50, 1500)

        TARGET_VIDEO_PATH = f"{HOME}/result.mp4"

        sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

        # create BYTETracker instance
        byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

        # create VideoInfo instance
        video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

        # create frame generator
        generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

        # create LineZone instance, it is previously called LineCounter class
        line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

        # create instance of BoxAnnotator
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

        # create instance of TraceAnnotator
        trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

        # create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
        line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

        # define call back function to be used in video processing
        def callback(frame: np.ndarray, index:int) -> np.ndarray:
            # model prediction on single frame and conversion to supervision Detections
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            # only consider class id from selected_classes define above
            detections = detections[np.isin(detections.class_id, selected_classes)]
            # tracking detections
            detections = byte_tracker.update_with_detections(detections)
            labels = [
                f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                for confidence, class_id, tracker_id
                in zip(detections.confidence, detections.class_id, detections.tracker_id)
            ]
            annotated_frame = trace_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            annotated_frame=box_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels)

            # update line counter
            line_zone.trigger(detections)
            # return frame with box and line annotated result
            return  line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

        # process the whole video
        sv.process_video(
            source_path = SOURCE_VIDEO_PATH,
            target_path = TARGET_VIDEO_PATH,
            callback=callback
        )

        # Dictionary to keep track of unique vehicles by their class and by minute
        vehicle_counts = collections.defaultdict(lambda: collections.defaultdict(set))

        # Define a function to get the timestamp in hours, minutes, seconds from the frame index
        def get_timestamp(frame_index: int, fps: int) -> str:
            # Convert frame index to seconds
            total_seconds = frame_index / fps
            # Get hours, minutes, seconds from total_seconds
            return str(dt.timedelta(seconds=total_seconds))
        
        # Define a callback function to process the frames
        def callback_with_time_count(frame: np.ndarray, index: int, fps: int) -> np.ndarray:
            # Model prediction on a single frame and conversion to supervision Detections
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            # Only consider class ids from selected_classes
            detections = detections[np.isin(detections.class_id, selected_classes)]
            # Track detections
            detections = byte_tracker.update_with_detections(detections)

            labels = [
                f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id)
            ]

            annotated_frame = trace_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

             # Get the timestamp for this frame
            timestamp = get_timestamp(index, fps)

            # Update line counter and check which detections crossed the line
            line_zone.trigger(detections)
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                # If this detection has crossed the line for the first time, count it
                if tracker_id not in vehicle_counts[timestamp]["all"]:
                    # Add the unique tracker_id to the set associated with its class_id at this timestamp
                    vehicle_counts[timestamp]["all"].add(tracker_id)
                    vehicle_counts[timestamp][CLASS_NAMES_DICT[class_id]].add(tracker_id)

            # Return frame with box and line-annotated results
            return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
        
        # Process the whole video with the updated callback
        fps = 30  # example frames per second; adjust to your video
        sv.process_video(
            source_path=SOURCE_VIDEO_PATH,
            target_path=TARGET_VIDEO_PATH,
            callback=lambda frame, index: callback_with_time_count(frame, index, fps)
        )

        # Get the last timestamp from vehicle_counts to summarize
        last_timestamp = max(vehicle_counts.keys())

        # Extract counts for the final timestamp
        final_counts = vehicle_counts[last_timestamp]

        # Calculate the total counts by vehicle class
        final_total_cars = len(final_counts["car"])
        final_total_buses = len(final_counts["bus"])
        final_total_trucks = len(final_counts["truck"])
        final_total_motorcycles = len(final_counts["motorcycle"])
        final_total_vehicles = len(final_counts["all"])

        def classify_area(car_count, motorcycle_count, truck_count, bus_count):
            # Decision rules for area classification
            if car_count > motorcycle_count and car_count > truck_count and car_count > bus_count:
                return "Area with majority of middle to high economy class"
            elif motorcycle_count > car_count and motorcycle_count > truck_count and motorcycle_count > bus_count:
                return "Area with majority of middle to low economy class "
            elif truck_count > car_count and truck_count > motorcycle_count and truck_count > bus_count:
                return "Area with majority of industrial presence"
            elif bus_count > car_count and bus_count > motorcycle_count and bus_count > truck_count:
                return "Area with majority of tourism activity)"
            else:
                return "Area with mixed vehicle presence"

        # Determine the area classification based on the counts
        area_classification = classify_area(final_total_cars, final_total_motorcycles, final_total_trucks, final_total_buses)


        print(f"Final unique counts at {last_timestamp}:")
        print("  Total unique cars:", final_total_cars)
        print("  Total unique buses:", final_total_buses)
        print("  Total unique trucks:", final_total_trucks)
        print("  Total unique motorcycles:", final_total_motorcycles)
        print("  Total unique vehicles:", final_total_vehicles)
        print("Area classification:", area_classification)

        analytics = await send_data(final_total_cars, final_total_buses, final_total_trucks, final_total_motorcycles, area_classification)
        
        # Remove video 
        os.remove('test.mp4')
        os.remove('result.mp4')

        attach_result = await data_attach(record_data['record_id'], analytics.data.data.analytics.id)
        print("Attach Result:", attach_result)

        return "SUCCESS"

async def send_data(car, bus, truck, bike, decision):
    try:
        ANALYTICS_ENDPOINT = "http://tp.kenaja.id/v1/api/analytics"  # API Endpoint

        # Prepare data for sending to analytics endpoint
        analytics_payload = {
            'decision': decision,
            'bikeCount': bike,
            'carCount': car,
            'truckCount': truck,
            'busCount' : bus
        }

        # Send data to the analytics endpoint
        response_analytics = requests.post(ANALYTICS_ENDPOINT, json=analytics_payload)
        if response_analytics.status_code != 200:
            return jsonify(response_analytics), 500
        
        # Parse the JSON response from the analytics endpoint
        analytics_response = response_analytics.json()

        return jsonify({'message': 'Data sent successfully.','data': analytics_response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


async def data_attach(record_id, analytics_id):
    try:
        ANALYTICS_ENDPOINT = "http://tp.kenaja.id/v1/api/records/attach"  # API Endpoint

        # Prepare data for sending to analytics endpoint
        attach_payload = {
            'recordId': record_id,
            'analyticsId': analytics_id,
        }

        # Send data to the analytics endpoint
        response_analytics = requests.post(ANALYTICS_ENDPOINT, json=attach_payload)
        if response_analytics.status_code != 200:
            return jsonify(response_analytics), 500
        
        # Parse the JSON response from the analytics endpoint
        analytics_response = response_analytics.json()

        return jsonify({'message': 'Data sent successfully.','data': analytics_response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)