
import cv2
import numpy as np
import os

def convert_rotated_annotation_line(line):
    """
    Convert a single rotated annotation (with angle) to an axis-aligned YOLO format.
    Expected input format (space-separated):
      class x_center y_center width height angle
    Returns a string in YOLO format:
      class x_center y_center width height
    All values are assumed to be absolute pixel values.
    """
    parts = line.strip().split()
    if len(parts) != 6:
        print("Warning: Expected 6 elements, got {} in line: {}".format(len(parts), line))
        return None
    
    cls = parts[0]
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    angle = float(parts[5])
    
    # Create the rotated rectangle: ((x_center, y_center), (width, height), angle)
    rect = ((x_center, y_center), (width, height), angle)
    # Get the 4 corner points of the rotated rectangle
    box = cv2.boxPoints(rect)  # 4x2 array
    # Compute minimal axis-aligned bounding box (i.e., the enclosing rectangle)
    x1 = np.min(box[:, 0])
    y1 = np.min(box[:, 1])
    x2 = np.max(box[:, 0])
    y2 = np.max(box[:, 1])
    
    # Convert to YOLO format: center_x, center_y, bbox_width, bbox_height
    new_x_center = (x1 + x2) / 2.0
    new_y_center = (y1 + y2) / 2.0
    new_width = x2 - x1
    new_height = y2 - y1
    
    # Return formatted string
    return f"{cls} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n"

def convert_rotated_annotations_in_file(input_file, output_file):
    """
    Reads a single annotation file with rotated annotations and writes a new file with
    axis-aligned YOLO annotations.
    """
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            converted_line = convert_rotated_annotation_line(line)
            if converted_line:
                fout.write(converted_line)
    print(f"Converted {input_file} -> {output_file}")

def convert_rotated_annotations_in_folder(input_folder, output_folder):
    """
    Processes all .txt annotation files in the input_folder, converts them to axis-aligned format,
    and saves them in output_folder. The output filenames will be the same as the input ones.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    if not files:
        print("No annotation files found in:", input_folder)
        return

    for file in files:
        input_file = os.path.join(input_folder, file)
        output_file = os.path.join(output_folder, file)
        convert_rotated_annotations_in_file(input_file, output_file)

if __name__ == "__main__":
    # Specify your folders.
    input_folder = "/Users/allenpereira/Desktop/d33ewd 2/datasets/labels/train"   # Folder containing your rotated annotation files.
    output_folder = "/Users/allenpereira/Desktop/d33ewd 2/datasets/labels/val2"  # Folder where axis-aligned annotations will be saved.

    convert_rotated_annotations_in_folder(input_folder, output_folder)
