import os

def get_station_paths(data_path):
    station_paths_list = []
    for patient_folder in sorted(os.listdir(data_path)):
        patient_path = os.path.join(data_path, patient_folder)
        if os.path.isdir(patient_path):
            station_paths_list.append([])
            for station_folder in os.listdir(patient_path):
                station_path = os.path.join(patient_path, station_folder)
                if os.path.isdir(station_path):
                    station_paths_list[-1].append(station_path)

    return station_paths_list

def get_frame_paths(station_path):
    frame_paths_list = []
    frame_names = os.listdir(station_path)
    sorted_frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1].replace('.png', '')))
    for frame in sorted_frame_names:
        frame_path = os.path.join(station_path, frame)
        if os.path.isfile(frame_path):
            frame_paths_list.append(frame_path)
    return frame_paths_list