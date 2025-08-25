import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load the YOLOv8 model
model_path = "models/hemletYoloV8_100epochs.pt"
model = YOLO(model_path)

# Load the test image with proper path handling
test_image_path = "test_images/Screenshot 2025-08-02 at 12.47.45 PM.png"
output_filename = "result_enhanced_Screenshot_2025-08-02_at_12.47.45_PM.png"
image = cv2.imread(test_image_path)

if image is None:
    print(f"Error: Could not load image at {test_image_path}")
    exit(1)

print(f"Successfully loaded image: {test_image_path}")

# Run YOLOv8 inference
results = model(image)[0]  # Get the first (and only) result

# Get class names from the model
class_names = model.names

# Create a custom color map for different confidence levels
def get_annotation_color(confidence):
    if confidence > 0.7:
        return (0, 255, 0)  # Green for high confidence
    elif confidence > 0.4:
        return (0, 191, 255)  # Orange for medium confidence
    else:
        return (0, 0, 255)  # Red for low confidence

# Create a new image with enhanced annotations
annotated_image = image.copy()

# Process detections
print("\nDetection Summary:")
for i, det in enumerate(results.boxes.data):
    x1, y1, x2, y2, confidence, class_id = map(float, det)
    class_id = int(class_id)
    class_name = results.names[class_id]
    
    # Print detection info
    print(f"{i+1}. {class_name} with confidence {confidence:.2f}")
    
    # Skip if confidence is too low
    if confidence < 0.25:  # You can adjust this threshold
        continue
        
    # Get color based on confidence
    color = get_annotation_color(confidence)
    
    # Convert coordinates to integers
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    # Create label with class and confidence
    label = f"{class_name} {confidence:.2f}"
    
    # Draw the bounding box
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
    
    # Add a filled background for the label
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(annotated_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
    
    # Add the text label
    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add a detection number for reference
    cv2.putText(annotated_image, f"#{i+1}", (x1, y2 + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# Save the result
output_path = f"test_images/{output_filename}"
cv2.imwrite(output_path, annotated_image)
print(f"\nDetection complete! Result saved to: {output_path}")

# Show the result
cv2.imshow("Helmet Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
