import cv2
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

# Load YOLOv5 model
weights = r'C:\ML_cource\Project\models\best_200_ep.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = attempt_load(weights, device=device)

# Define class labels
class_labels = ['cabbage', 'weed']

# Initialize SORT tracker
class SortTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        # Update existing tracks with new detections
        for track_id, track in self.tracks.items():
            if track['time_since_update'] > 0:
                track['time_since_update'] -= 1

        for detection in detections:
            x_min, y_min, x_max, y_max, conf, cls_id = detection
            cls_name = class_labels[int(cls_id)]
            best_match_id = None
            best_iou = 0

            for track_id, track in self.tracks.items():
                if track['time_since_update'] == 0:
                    track_box = track['box']
                    iou = self.compute_iou((x_min, y_min, x_max, y_max), track_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = track_id

            if best_match_id is not None and best_iou > 0.5:
                self.tracks[best_match_id]['box'] = (x_min, y_min, x_max, y_max)
                self.tracks[best_match_id]['time_since_update'] = 0
            else:
                self.tracks[self.next_id] = {
                    'box': (x_min, y_min, x_max, y_max),
                    'time_since_update': 0,
                    'class': cls_name
                }
                self.next_id += 1

        # Remove old tracks
        self.tracks = {track_id: track for track_id, track in self.tracks.items()
                       if track['time_since_update'] < 10}

        # Get tracked objects for visualization
        tracked_objects = []
        for track_id, track in self.tracks.items():
            tracked_objects.append(track['box'] + (track_id, track['class']))

        return tracked_objects

    @staticmethod
    def compute_iou(box1, box2):
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2

        xA = max(x_min1, x_min2)
        yA = max(y_min1, y_min2)
        xB = min(x_max1, x_max2)
        yB = min(y_max1, y_max2)

        intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        box1_area = (x_max1 - x_min1 + 1) * (y_max1 - y_min1 + 1)
        box2_area = (x_max2 - x_min2 + 1) * (y_max2 - y_min2 + 1)

        iou = intersection_area / float(box1_area + box2_area - intersection_area)

        return iou


# Initialize SORT tracker
tracker = SortTracker()

# Open the video file
video_path = r'C:\Users\vlado\OneDrive\Desktop\my_video_12.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection using YOLOv5
    img = torch.from_numpy(frame).to(device)
    img = img.float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    results = model(img)[0]
    results = non_max_suppression(results, conf_thres=0.3, iou_thres=0.45)

    # Prepare detections for SORT
    detections = []
    for result in results:
        if result is not None and len(result) > 0:
            for det in result:
                x_min, y_min, x_max, y_max, conf, cls_id = det
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                detections.append([x_min, y_min, x_max, y_max, conf, cls_id])

    # Pass the detections to SORT for tracking
    if len(detections) > 0:
        tracked_objects = tracker.update(detections)

        # Draw bounding boxes and labels on the frame
        for obj in tracked_objects:
            x_min, y_min, x_max, y_max, track_id, cls_name = obj
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {track_id}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with tracked objects
    cv2.imshow('Frame', frame)
    
    
    print(f'Tracker: {tracker}')

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
# cv2.destroyAllWindows()
