import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_type):
        self.model_type = model_type
        self.net = None
        self.classes = []
        self.colors = []

    def load_model(self):
        # if self.model_type == 'effdet':
        #     # Load EfficientDet model and configuration files
        #     config_file = 'path/to/effdet/config'
        #     weights_file = 'path/to/effdet/weights'
        #     self.net = cv2.dnn.readNetFromTensorflow(weights_file, config_file)

            # Load class labels and corresponding colors
            # self.classes = ['class1', 'class2', 'class3', ...]
            # self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        #elif self.model_type == 'yolo':
            # Load YoloV2 model and configuration files
            config_file = 'yolov2.cfg'
            weights_file = 'yolov2.weights'
            self.net = cv2.dnn.readNetFromDarknet(config_file, weights_file)

            # Load class labels and corresponding colors
            with open('classes.txt') as f:
                self.classes = [line.strip() for line in f]
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect_objects(self, frame):
        # Convert frame to blob for input to neural network
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

        # Pass blob through neural network
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        # Parse object detection results and draw boxes
        boxes = []
        confidences = []
        class_ids = []
        height, width = frame.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw boxes and labels for each detected object
        for i in indices:
            x, y, w, h = boxes[i]
            color = self.colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f'{self.classes[class_ids[i]]}: {confidences[i]:.2f}'
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


class VideoPlayer:
    def __init__(self, video_path, detector1):
        self.video_path = video_path
        self.detector1 = detector1
        #self.detector2 = detector2

    def play_video(self):
        # Open video file
        cap = cv2.VideoCapture(self.video_path)

        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create two OpenCV windows for displaying object detection results
        cv2.namedWindow('Detector 1', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('Detector 2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detector 1', width // 2, height)
        #cv2.resizeWindow('Detector 2', width // 2, height)

        # Initialize variables for tracking progress
        frame_num = 0

        while True:
            # Read next frame from video file
            ret, frame = cap.read()

            if not ret:
                break

            # Apply object detection to frame using both detectors
            frame1 = self.detector1.detect_objects(frame.copy())
            #frame2 = self.detector2.detect_objects(frame.copy())

            # Show object detection results in separate windows
            cv2.imshow('Detector 1', frame1)
            #cv2.imshow('Detector 2', frame2)


            # Exit if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Update progress
            frame_num += 1
            print(f'Processed frame {frame_num}/{total_frames} ({frame_num / total_frames:.2%})')

        # Release video file and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

#Create instances of ObjectDetector for each model
#effdet_detector = ObjectDetector('effdet')
yolo_detector = ObjectDetector('yolo')

#Load models for each detector
#effdet_detector.load_model()
yolo_detector.load_model()

#Create instance of VideoPlayer and play video
video_path = 'video.mp4'
player = VideoPlayer(video_path, yolo_detector)
player.play_video()