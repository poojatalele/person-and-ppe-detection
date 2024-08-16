import os
import cv2
import argparse

# Ensure the output directories exist
def ensure_directories_exist(*dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

# Calculate Euclidean distance between two points.
def calculate_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def crop_persons_and_adjust_ppe_labels(person_labels_dir, ppe_labels_dir, ppe_images_dir, cropped_images_dir, cropped_labels_dir):
    ensure_directories_exist(cropped_images_dir, cropped_labels_dir)

    for label_file in os.listdir(person_labels_dir):
        person_label_path = os.path.join(person_labels_dir, label_file)
        ppe_label_path = os.path.join(ppe_labels_dir, label_file)
        image_path = os.path.join(ppe_images_dir, label_file.replace('.txt', '.jpg'))

        # Read the person label file
        with open(person_label_path, 'r') as file:
            person_lines = file.readlines()

        # Load the corresponding image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]

        # Track the number of persons in the image
        person_count = 0

        for line in person_lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            
            # Process the label if it is for a person (class_id == 0)
            if class_id == 0:
                person_count += 1
                
                # Extract the person's bounding box information
                x_center, y_center, width, height = map(float, parts[1:])
                xmin = int((x_center - width / 2) * img_width)
                xmax = int((x_center + width / 2) * img_width)
                ymin = int((y_center - height / 2) * img_height)
                ymax = int((y_center + height / 2) * img_height)

                # Ensure bounding box is within image bounds
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(img_width, xmax), min(img_height, ymax)

                # Crop the person from the image
                cropped_image = image[ymin:ymax, xmin:xmax]

                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    cropped_image_path = os.path.join(cropped_images_dir, f'{os.path.splitext(label_file)[0]}_person_{person_count}.jpg')
                    cv2.imwrite(cropped_image_path, cropped_image)

                    # Read the corresponding PPE label file
                    with open(ppe_label_path, 'r') as ppe_file:
                        ppe_lines = ppe_file.readlines()

                    adjusted_lines = []
                    ppe_distances = []
                    for ppe_line in ppe_lines:
                        ppe_parts = ppe_line.strip().split()
                        ppe_class_id = int(ppe_parts[0])

                        if ppe_class_id >= 0:  # Adjust labels
                            ppe_x_center, ppe_y_center, ppe_width, ppe_height = map(float, ppe_parts[1:])
                            ppe_xmin = (ppe_x_center - ppe_width / 2) * img_width
                            ppe_xmax = (ppe_x_center + ppe_width / 2) * img_width
                            ppe_ymin = (ppe_y_center - ppe_height / 2) * img_height
                            ppe_ymax = (ppe_y_center + ppe_height / 2) * img_height

                            # Adjust the bounding box for the cropped region
                            new_xmin = max(ppe_xmin - xmin, 0)
                            new_ymin = max(ppe_ymin - ymin, 0)
                            new_xmax = min(ppe_xmax - xmin, xmax - xmin)
                            new_ymax = min(ppe_ymax - ymin, ymax - ymin)

                            new_width = (new_xmax - new_xmin) / (xmax - xmin)
                            new_height = (new_ymax - new_ymin) / (ymax - ymin)
                            new_x_center = (new_xmin + new_xmax) / 2 / (xmax - xmin)
                            new_y_center = (new_ymin + new_ymax) / 2 / (ymax - ymin)

                            if new_width > 0 and new_height > 0:
                                distance_to_person_center = calculate_distance(new_x_center, new_y_center, 0.5, 0.5)
                                ppe_distances.append((distance_to_person_center, ppe_class_id, new_x_center, new_y_center, new_width, new_height))

                    # Sort PPE by distance to the center of the person
                    ppe_distances.sort(key=lambda x: x[0])

                    for _, ppe_class_id, new_x_center, new_y_center, new_width, new_height in ppe_distances:
                        adjusted_line = f"{ppe_class_id} {new_x_center} {new_y_center} {new_width} {new_height}"
                        adjusted_lines.append(adjusted_line)

                    if adjusted_lines:
                        cropped_label_path = os.path.join(cropped_labels_dir, f'{os.path.splitext(label_file)[0]}_person_{person_count}.txt')
                        with open(cropped_label_path, 'w') as f:
                            f.write('\n'.join(adjusted_lines) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop persons from images and adjust PPE labels.")
    parser.add_argument('--person_labels_dir', type=str, required=True, help="Directory containing person detection labels.")
    parser.add_argument('--ppe_labels_dir', type=str, required=True, help="Directory containing PPE detection labels.")
    parser.add_argument('--ppe_images_dir', type=str, required=True, help="Directory containing the images.")
    parser.add_argument('--cropped_images_dir', type=str, required=True, help="Directory to save cropped images.")
    parser.add_argument('--cropped_labels_dir', type=str, required=True, help="Directory to save adjusted labels.")
    
    args = parser.parse_args()
    
    crop_persons_and_adjust_ppe_labels(
        person_labels_dir=args.person_labels_dir,
        ppe_labels_dir=args.ppe_labels_dir,
        ppe_images_dir=args.ppe_images_dir,
        cropped_images_dir=args.cropped_images_dir,
        cropped_labels_dir=args.cropped_labels_dir
    )
