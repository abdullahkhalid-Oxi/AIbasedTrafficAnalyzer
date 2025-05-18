import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Load the image
image_path = 'morecars.JPG'
image = cv2.imread(image_path)

# Detect objects
results = model(image)

# List of vehicle classes YOLOv8 can detect (based on COCO dataset)
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

# Draw boxes and labels for all vehicles
for result in results[0].boxes.data:
    x1, y1, x2, y2, confidence, class_id = result.tolist()
    if confidence > 0:  # Only show confident detections
        class_name = model.names[int(class_id)]
        if class_name in vehicle_classes:  # Check if it's a vehicle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'{class_name} {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


# Optional: Count vehicle types
vehicle_counts = {}
for result in results[0].boxes.data:
    x1, y1, x2, y2, confidence, class_id = result.tolist()
    if confidence > 0:
        class_name = model.names[int(class_id)]
        if class_name in vehicle_classes:
            vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1

# Print counts
print("-----------------Vehicle Counts--------------------")
for vehicle, count in vehicle_counts.items():
    print(f"{vehicle}: {count}")

# Save and show the result
cv2.imwrite('output_img.JPG', image)
print("Saved result to output_image1.JPG")
cv2.imshow('Detected Vehicles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()