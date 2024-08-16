import os
import argparse

def separate_labels(input_dir, person_output_dir, ppe_output_dir, person_class=0):
    
    # Ensure the output directories exist
    os.makedirs(person_output_dir, exist_ok=True)
    os.makedirs(ppe_output_dir, exist_ok=True)

    for label_file in os.listdir(input_dir):
        input_path = os.path.join(input_dir, label_file)

        with open(input_path, 'r') as file:
            lines = file.readlines()

        person_lines = []
        ppe_lines = []

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])

            if class_id == person_class:
                person_lines.append(line)
            else:
                # Adjust class ID for non-person classes
                adjusted_class_id = class_id - 1
                adjusted_line = f"{adjusted_class_id} " + " ".join(parts[1:]) + "\n"
                ppe_lines.append(adjusted_line)

        # Save the person labels
        if person_lines:
            person_output_path = os.path.join(person_output_dir, label_file)
            with open(person_output_path, 'w') as file:
                file.writelines(person_lines)

        # Save the PPE labels with adjusted class IDs
        if ppe_lines:
            ppe_output_path = os.path.join(ppe_output_dir, label_file)
            with open(ppe_output_path, 'w') as file:
                file.writelines(ppe_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate person labels from other labels and save them into different directories with adjusted class IDs for PPE.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing the original labels.")
    parser.add_argument('--person_output_dir', type=str, required=True, help="Directory to save person labels.")
    parser.add_argument('--ppe_output_dir', type=str, required=True, help="Directory to save PPE labels.")
    parser.add_argument('--person_class', type=int, default=0, help="Class ID for the person (default is 0).")
    
    args = parser.parse_args()
    
    separate_labels(
        input_dir=args.input_dir,
        person_output_dir=args.person_output_dir,
        ppe_output_dir=args.ppe_output_dir,
        person_class=args.person_class
    )
