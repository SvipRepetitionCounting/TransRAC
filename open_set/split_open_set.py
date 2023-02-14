import numpy as np
import os
def get_frame(root_dir):
    total_num = 0
    for curdis, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("npz"):
                data_path = os.path.join(curdis, file)
                with np.load(data_path) as data:
                    total_num += data['fps'].item()
    print(root_dir.split("/")[1], total_num)


get_frame("./train")
get_frame("./valid")
get_frame("./test")
