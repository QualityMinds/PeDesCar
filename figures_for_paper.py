import json
from pathlib import Path

# import pedestrian_impact as env
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

experiment_path = "logs/pedestrian_collision_dataset/"
path = Path(experiment_path)
files = list(Path(experiment_path).joinpath("numpy").glob("*.npy"))
files.sort()
dataset = []
for i, file in enumerate(files):
    dataset.append(np.load(file, allow_pickle=True))
    print(f'{i / len(files) * 100:.2f}', end='\r', flush=True)

files = list(Path(experiment_path).joinpath("files").glob("*.json"))
files.sort()
metas = []
for i, file in enumerate(files):
    metas.append(json.load(open(file))["meta"])
    print(f'{i / len(files) * 100:.2f}', end='\r', flush=True)

joint_names = np.load(Path(experiment_path).joinpath("body.npy"))
# get initial car and pedestrian positions
car_idx = np.where("car" == joint_names)[0][0]
human_idx = np.where("pelvis" == joint_names)[0][0]

init_car_pos = []
init_human_pos = []
speeds = []
car_model = []
for d, m in zip(dataset, metas):
    car_model.append(m["data"]["meta_data_full"][0]["car_type"])
    init_car_pos.append(d[0, car_idx])
    init_human_pos.append(d[0, human_idx])
    speeds.append(m["data"]["meta_data_full"][0]["collision_speed (km/h)"])
init_car_pos = np.array(init_car_pos)
init_human_pos = np.array(init_human_pos)
speeds = np.array(speeds)
car_model = np.uint8("autocar" == np.array(car_model)).flatten()
interp_func = scipy.interpolate.interp1d([speeds.min(), speeds.max()], [0.1, 50.], kind='linear')
speeds_plot = interp_func(speeds)

col = np.array(["r", "g"])
plt.figure(figsize=(12, 12))
plt.scatter(init_car_pos[car_model == 0, 1], init_car_pos[car_model == 0, 0], s=speeds[car_model == 0], c=col[0],
            label="prius car")
plt.scatter(init_car_pos[car_model == 1, 1], init_car_pos[car_model == 1, 0], s=speeds[car_model == 1], c=col[1],
            label="auto car")
plt.scatter(init_human_pos[:, 1], init_human_pos[:, 0], s=5., c="b", label="human")
plt.title("Initial Position: Human vs Cars (speed scaled in car)")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid()
plt.legend()

plt.figure(figsize=(12, 12))
plt.hist(speeds, bins=50, density=True)
plt.title("Collision speeds Histogram")
plt.xlabel("Speeds")
plt.ylabel("Density")
plt.grid()

# get simulation time
simulation_times = [m["data"]["meta_data_full"][0]["simulation_time"] for m in metas]
plt.figure(figsize=(12, 12))
weights = np.ones(len(simulation_times)) / len(simulation_times)
plt.hist(simulation_times, bins=50, weights=weights)
plt.title("Simulation times Histogram")
plt.xlabel("Seconds")
plt.ylabel("Density")
plt.grid()

sim_time = [len(d) for d in dataset]
plt.figure(figsize=(12, 12))
weights = np.ones(len(sim_time)) / len(sim_time)
plt.hist(sim_time, bins=50, weights=weights)
plt.title("Simulation frame length Histogram")
plt.xlabel("Simulation frames")
plt.ylabel("Density")
plt.grid()

car_model = [m["data"]["meta_data_full"][0]["car_type"] for m in metas]
plt.figure(figsize=(12, 12))
weights = np.ones(len(car_model)) / len(car_model)
plt.hist(car_model, bins=2, weights=weights)
plt.title("Car model Histogram")
plt.xlabel("Car model")
plt.ylabel("Density")
plt.grid()
plt.show()

activity = [m["data"]["meta_data_full"][0]["activity_type"] for m in metas]
plt.figure(figsize=(12, 12))
weights = np.ones(len(activity)) / len(activity)
plt.hist(activity, bins=2, weights=weights)
plt.title("Simulation vs Activity Histogram")
plt.xlabel("Activity")
plt.ylabel("Density")
plt.grid()

# humanoid_mode
