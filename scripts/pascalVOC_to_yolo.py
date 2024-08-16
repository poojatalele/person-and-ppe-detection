import xml.etree.ElementTree as ET
import os
import argparse

classes = ['person', 'hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 'ppe-suit', 'ear-protector', 'safety-harness']

# Convert bounding box to YOLO format.
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

# Convert a single annotation file from Pascal VOC to YOLO format.
def convert_annotation(image_id, voc_dir, yolo_dir, classes):
    in_file = open(os.path.join(voc_dir, 'labels', f'{image_id}.xml'))
    out_file = open(os.path.join(yolo_dir, f'{image_id}.txt'), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        bounding_box = (
            float(xmlbox.find('xmin').text), 
            float(xmlbox.find('xmax').text), 
            float(xmlbox.find('ymin').text), 
            float(xmlbox.find('ymax').text)
        )
        yolo_bounding_box = convert((image_width, image_height), bounding_box)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in yolo_bounding_box]) + '\n')

def main(voc_dir, yolo_dir):
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)

    for image_filename in os.listdir(os.path.join(voc_dir, 'images')):
        image_id = os.path.splitext(image_filename)[0]
        convert_annotation(image_id, voc_dir, yolo_dir, classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Pascal VOC annotations to YOLO format.')
    parser.add_argument('--voc_dir', type=str, required=True, help='Path to the VOC dataset directory.')
    parser.add_argument('--yolo_dir', type=str, required=True, help='Path to save the YOLO formatted annotations.')

    args = parser.parse_args()

    main(args.voc_dir, args.yolo_dir)
