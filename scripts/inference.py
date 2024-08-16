import os
import cv2
import argparse
from ultralytics import YOLO

def run_inference(image_path, output_dir, person_model, ppe_model):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        return
    
    img_height, img_width = image.shape[:2]
    
    # Run person detection
    person_results = person_model(image)
    
    for result in person_results:
        for bbox in result.boxes.xyxy:
            
            # Extract bounding box coordinates
            xmin, ymin, xmax, ymax = map(int, bbox)
            
            # Ensure coordinates are within image bounds
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(img_width, xmax), min(img_height, ymax)
            
            # Crop the person image
            cropped_image = image[ymin:ymax, xmin:xmax]
            
            if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                # Run PPE detection on the cropped image
                ppe_results = ppe_model(cropped_image)
                
                for ppe_result in ppe_results:
                    for ppe_bbox, ppe_class_id in zip(ppe_result.boxes.xyxy, ppe_result.boxes.cls):
                        ppe_xmin, ppe_ymin, ppe_xmax, ppe_ymax = map(int, ppe_bbox)
                        
                        # Adjust PPE bounding box coordinates relative to the original image
                        ppe_xmin += xmin
                        ppe_ymin += ymin
                        ppe_xmax += xmin
                        ppe_ymax += ymin
                        
                        # Ensure adjusted coordinates are within the original image bounds
                        ppe_xmin, ppe_ymin = max(0, ppe_xmin), max(0, ppe_ymin)
                        ppe_xmax, ppe_ymax = min(img_width, ppe_xmax), min(img_height, ppe_ymax)
                        
                        # Draw bounding box for PPE detection on the original image
                        cv2.rectangle(image, (ppe_xmin, ppe_ymin), (ppe_xmax, ppe_ymax), (0, 255, 0), 2)

                        # Annotate the image with the class ID
                        cv2.putText(image, str(int(ppe_class_id)), (ppe_xmin, ppe_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the image with detections
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Saved inference result to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run inference using person and PPE detection models.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing images for inference")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save inference images")
    parser.add_argument('--person_det_model', type=str, required=True, help="Path to the person detection model")
    parser.add_argument('--ppe_detection_model', type=str, required=True, help="Path to the PPE detection model")
    
    args = parser.parse_args()

    # Load the models
    person_model = YOLO(args.person_det_model)
    ppe_model = YOLO(args.ppe_detection_model)

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate over images in the directory and run inference
    for image_file in os.listdir(args.input_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(args.input_dir, image_file)
            run_inference(image_path, args.output_dir, person_model, ppe_model)

if __name__ == "__main__":
    main()
