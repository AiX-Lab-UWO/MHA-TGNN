import os
import csv
import mediapipe as mp
import cv2
import time

start_time = time.time()

# Initialize mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)

# Directories
train_dir = r'D:\DGW\...'

# CSV and log file paths (save in D:\DGW)
train_csv_path = r'D:\DGW\...csv'
train_log_path = r'D:\...txt'

# Function to extract landmarks and save them in a CSV
def process_images(directory, csv_file, log_file):
    with open(csv_file, mode='w', newline='') as csv_f, open(log_file, mode='w') as log_f:
        writer = csv.writer(csv_f)

        # Correct header: filename, x0, y0, z0, x1, y1, z1, ..., x477, y477, z477, Label
        header = ['filename'] + [f'x{i}' for i in range(478)] + [f'y{i}' for i in range(478)] + [f'z{i}' for i in range(478)] + ['Label']
        writer.writerow(header)

        # Go through each main folder (1 to 9)
        for subfolder in range(1, 10):
            base_folder_path = os.path.join(directory, str(subfolder))

            # Process original folder
            if os.path.exists(base_folder_path):
                process_folder_images(base_folder_path, subfolder, writer, log_f)

            # Process augmented folders
            for aug_type in ["blurred", "contrast", "gaussian_noise", "histogram_equalized", "intensity_dark", "intensity_light", "sharpened"]:
                aug_folder_path = os.path.join(directory, f"{subfolder}_{aug_type}")
                if os.path.exists(aug_folder_path):
                    process_folder_images(aug_folder_path, subfolder, writer, log_f)


def process_folder_images(folder_path, label, writer, log_f):
    """Process all images in a given folder and write their landmarks to the CSV."""
    for img_name in os.listdir(folder_path):
        if img_name.endswith('.png'):
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                log_f.write(f"{img_name}\n")
                continue

            # Convert the image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe
            result = face_mesh.process(rgb_image)

            # Check if landmarks are found
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark

                # Organize landmarks in the correct order: x0, y0, z0, x1, y1, z1, ...
                row = [img_name]  # Start with filename
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z])  # Append x, y, z for each landmark

                # Add label corresponding to the original folder number (1 to 9)
                row.append(label)

                writer.writerow(row)
            else:
                # Log failed images
                log_f.write(f"{img_name}\n")

# Process training images
process_images(train_dir, train_csv_path, train_log_path)

# Close MediaPipe instance
face_mesh.close()

print("Landmark extraction completed.")

end_time = time.time()

print(f'Landmark extraction time: {end_time - start_time:.2f} seconds')
