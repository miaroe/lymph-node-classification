import os
from datetime import datetime

import os
from datetime import datetime

def get_latest_date_time(directory_path):
    latest_date = None
    latest_time = None

    for root, dirs, files in os.walk(directory_path):
        for dir_name in dirs:
            # Extract date from directory name
            try:
                date = datetime.strptime(dir_name, "%Y-%m-%d").date()
            except ValueError:
                continue  # Skip directories that don't match the expected date format

            if latest_date is None or date > latest_date:
                latest_date = date
                latest_time = None  # Reset latest_time whenever a new latest_date is found

            if date == latest_date:
                # Traverse subdirectories and find the latest time within the latest date folder
                subdirectory_path = os.path.join(root, dir_name)
                for sub_root, sub_dirs, sub_files in os.walk(subdirectory_path):
                    for sub_dir_name in sub_dirs:
                        # Extract time from subdirectory name
                        try:
                            time = datetime.strptime(sub_dir_name, "%H:%M:%S").time()
                        except ValueError:
                            continue  # Skip subdirectories that don't match the expected time format

                        if latest_time is None or time > latest_time:
                            latest_time = time

    if latest_date is not None and latest_time is not None:
        latest_date_time = f"{latest_date}/{latest_time}"
        return latest_date_time

    return None

