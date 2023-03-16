import json

# Define the classes and output file name
classes = ["cabbage", "weed"]

# Define path to Vott json file:
vott_json_path = '../data/labeled_output/vott_labels.json'

# Define path to YOLO output file:
yolo_output_file = "../data/labeled_output/YOLO_labels.txt"

# Load the JSON file
with open(vott_json_path) as f:
    data = json.load(f)

# Loop through the frames and write the bounding boxes to the output file
with open(yolo_output_file, "w") as f:
    for frame in data["frames"]:
        for bbox in data["frames"][frame]:
            class_id = classes.index(bbox["tags"][0])
            x_center = (bbox["x1"] + bbox["x2"]) / 2 / bbox["width"]
            y_center = (bbox["y1"] + bbox["y2"]) / 2 / bbox["height"]
            width = (bbox["x2"] - bbox["x1"]) / bbox["width"]
            height = (bbox["y2"] - bbox["y1"]) / bbox["height"]
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
