import cv2
import os
import json
import re

# Path to the video
video_path = r"F:\DMD\S6_face_RGB\7\gB_7_s6_2019-03-11T14;08;04+01;00_ir_face.mp4"

# Path to the gaze zone JSON file
json_file_path = r'F:\DMD\S6_face_RGB\7\gB_7_s6_2019-03-11T14;08;04+01;00_rgb_ann_gaze.json'

# Directory to save valid frames
output_folder = os.path.join(os.path.dirname(video_path), "Frames_IR_Valid")

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load gaze zone data from JSON
with open(json_file_path, 'r') as file:
    gaze_data = json.load(file)

# Mapping of gaze zones to their respective names
gaze_zone_names = {
    0: 'left_mirror',
    1: 'left',
    2: 'front',
    3: 'center_mirror',
    4: 'front_right',
    5: 'right_mirror',
    6: 'right',
    7: 'infotainment',
    8: 'steering_wheel',
    9: 'not_valid'  # Default value if no gaze zone is found
}

# Extract valid frames from the JSON data
actions = gaze_data.get('openlabel', {}).get('actions', {})
valid_frames = set()
for action_info in actions.values():
    if 'gaze_zone' in action_info.get('type', ''):
        gaze_zone = action_info['type'].split('/')[-1]
        if gaze_zone != 'not_valid':
            for interval in action_info.get('frame_intervals', []):
                valid_frames.update(range(interval['frame_start'], interval['frame_end'] + 1))

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    # If the frame is not read correctly, end the loop
    if not ret:
        break

    # Save only valid frames
    if frame_count in valid_frames:
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Valid frames have been saved to {output_folder}.")


###################Landmark Extraction
import mediapipe as mp
import cv2
import csv
import os
import json
import re
import time

# Initialize Mediapipe Face Mesh model with refined landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def sorted_frame_files(frames_dir):
    frame_files = os.listdir(frames_dir)
    frame_files = [f for f in frame_files if f.startswith('frame_') and f.endswith('.jpg')]
    return sorted(frame_files, key=lambda x: int(re.search(r'(\d+)', x).group()))


def extract_gaze_data_and_landmarks(json_file_path, frames_dir, output_csv_path, failed_frames_file_path):
    with open(json_file_path, 'r') as file:
        gaze_data = json.load(file)

    # Mapping of gaze zones to their respective names
    gaze_zone_names = {
        0: 'left_mirror',
        1: 'left',
        2: 'front',
        3: 'center_mirror',
        4: 'front_right',
        5: 'right_mirror',
        6: 'right',
        7: 'infotainment',
        8: 'steering_wheel',
        9: 'not_valid'  # Default value if no gaze zone is found
    }

    # Extract all actions with their frame intervals
    actions = gaze_data.get('openlabel', {}).get('actions', {})
    frame_gaze_mapping = {}
    for action_info in actions.values():
        if 'gaze_zone' in action_info.get('type', ''):
            gaze_zone = action_info['type'].split('/')[-1]
            gaze_zone_number = [num for num, name in gaze_zone_names.items() if name == gaze_zone][0]
            for interval in action_info.get('frame_intervals', []):
                for frame in range(interval['frame_start'], interval['frame_end'] + 1):
                    frame_gaze_mapping[frame] = gaze_zone_number

    failed_frames = []
    extraction_times = 0

    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        headers = ['Frame Index', 'Gaze Zone', 'Gaze Zone Number'] + ['landmark_{}'.format(i) for i in range(478 * 3)]
        writer.writerow(headers)

        sorted_frames = sorted_frame_files(frames_dir)
        for frame_file in sorted_frames:
            frame_index = int(re.search(r'(\d+)', frame_file).group())
            gaze_zone_number = frame_gaze_mapping.get(frame_index, 9)
            if gaze_zone_number == 9:
                continue

            full_frame_path = os.path.join(frames_dir, frame_file)
            image = cv2.imread(full_frame_path)
            if image is None:
                failed_frames.append(frame_file)
                continue

            with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:

                # Start timing landmark extraction
                start_time = time.time()

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_image)

                # End timing after extraction
                end_time = time.time()
                extraction_times = extraction_times + (end_time - start_time)  # Store extraction time

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    flattened_landmarks = [coord for landmark in landmarks for coord in (landmark.x, landmark.y, landmark.z)]

                    gaze_zone_name = gaze_zone_names.get(gaze_zone_number, 'not_valid')
                    writer.writerow([frame_index, gaze_zone_name, gaze_zone_number] + flattened_landmarks)
                else:
                    failed_frames.append(frame_file)
    print(extraction_times)
    with open(failed_frames_file_path, 'w') as f:
        for frame in failed_frames:
            f.write(f"{frame}\n")


# Define the file paths
# json_file_path = r'F:\DMD\S6_face_RGB\4\gA_3_s6_2019-03-08T10;12;10+01;00_rgb_ann_gaze.json'
frames_dir = r'F:\DMD\S6_face_RGB\7\Frames_IR_Valid'
output_csv_path = r'F:\DMD\S6_face_RGB\7\Landmark_IR_Valid.csv'
failed_frames_file_path = r'F:\DMD\S6_face_RGB\7\Failed_IR.txt'

# Run the function with the given file paths
extract_gaze_data_and_landmarks(json_file_path, frames_dir, output_csv_path, failed_frames_file_path)

print(f"landmarks have been saved.")
