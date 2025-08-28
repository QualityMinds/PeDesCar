import time

import mujoco
import numpy as np

import pedestrian_impact as env
from logger import Logger


def main(args):
    np.random.seed(args.seed)
    # env.kmh2ms(75) => 20.83
    # env.kmh2ms(10) => 2.78
    # record every 1ms
    wrapper = None
    logger = Logger(path=args.experiment_log)
    collision_simulation = 0  # Set init counter here.
    total_collisions_generated = int(args.total_collisions_generated)
    if args.init_collision:
        collision_simulation = args.init_collision
        total_collisions_generated += collision_simulation
    experiment_settings = collision_simulation - 1
    while collision_simulation < total_collisions_generated:
        experiment_settings += 1
        if experiment_settings % args.num_simulations_per_env == 0 or wrapper is None:
            envdet = {"activity": np.random.choice(args.activity_type),
                      "skeleton": np.random.choice(args.skeleton_type),
                      "car_type": np.random.choice(args.car_type),
                      "after_collision": np.random.choice(args.after_collision_type)
                      }
            print(envdet)
            # viewer_params = {} # Needed for testing Future visualization
            wrapper = env.HumanoidCollision(env_name=f'HumanCollision{envdet["skeleton"]}.{envdet["activity"]}',
                                            car_model=envdet["car_type"],
                                            agent_path=args.agent_path,
                                            use_foot_forces=args.use_foot_forces,
                                            use_muscles=False if envdet["skeleton"] != "Muscle" else True,
                                            disable_arms=False,
                                            timestep=args.timestep,
                                            n_substeps=args.n_substeps,
                                            seed=args.seed,
                                            # viewer_params=viewer_params,
                                            std=0.5)
            renderer = None
            if args.render is not None:  # tested on 2.3.6 and 3.1.3
                if not args.render:
                    renderer = mujoco.renderer.Renderer(wrapper.env._model, 480,
                                                        640)  # for bigger resolutions 1080, 1920
            logger.update_model(wrapper.env._model, task=envdet["activity"])

        done = False
        fall = False
        processing_time1 = time.process_time()
        activate_agent = True
        # init a new env with new qpos and qvel
        observation, meta_data = wrapper.init_env(np.float32(args.init_car_speed),
                                                  np.float32(args.init_car_turn),
                                                  np.float32(args.init_car_position),
                                                  np.float32(args.init_car_position_shift),
                                                  np.float32(args.init_human_position),
                                                  np.float32(args.init_human_orientation),
                                                  args.flip_human,
                                                  )
        logger.new_frame(collision_simulation)
        video_data = {}
        if renderer:
            for cam in args.cameras: video_data[cam] = []  # camera settings
        for it in range(int(2. / args.timestep)):
            if activate_agent:
                action = wrapper.agent.draw_action(observation[:-len(wrapper.env.obs_names)])
            else:
                if envdet["after_collision"] == "zero":
                    action = np.zeros(wrapper.env.info.action_space._shape[0] - 3)  # 3 comes from the car
                else:
                    action = np.random.uniform(wrapper.env.info.action_space.low[:-3] * 0.1,
                                               wrapper.env.info.action_space.high[:-3] * 0.1)
            actions_w_car = np.concatenate([action, [0, 15, 0]])
            try:
                if (observation[2] < -2. or
                        observation[2] > 5. or  # height
                        observation[0] < -20 or
                        observation[0] > 20 or  # X coordinate
                        observation[1] < -20 or
                        observation[1] > 20):  # Z coordinate
                    print(  # simulation break
                        f'sim:{experiment_settings} sim_iter:{it} - env:{envdet["skeleton"]}.{envdet["activity"]}  -  WEIRD SIMULATION VALUES')
                    break
                observation, reward, absorbing, info = wrapper.env.step(actions_w_car)
            except:
                print(
                    f'sim:{experiment_settings} sim_iter:{it} - env:{envdet["skeleton"]}.{envdet["activity"]}  -  SIMULATION ERROR')
                break
            if not np.round(it * args.timestep, 10) * 1000 % 10:
                logger.add_frame(it, wrapper.env._data)  # to record all frames remove the IF above

            if args.render is None:
                pass
            elif args.render:
                wrapper.env.render()
            else:  # Not validated in the newest mujoco versions =(
                if not np.round(it * args.timestep, 10) * 1000 % 10:
                    for cam in args.cameras:
                        renderer.update_scene(wrapper.env._data, camera=cam)
                        video_data[cam].append(renderer.render())

            if activate_agent:
                activate_agent, collision_speed = wrapper.verify_collision_w_car()
                collision_it = it
            if absorbing:
                if not activate_agent:
                    done = True
                else:
                    fall = True
                break
            elif it > 1.25 / args.timestep and not activate_agent:
                done = True
            elif it > 1.25 / args.timestep and activate_agent:
                logger.clean_data()  # PATCH to remove data
                video_data.clear()  # PATCH to remove data
                break
        if done:
            print(
                f'sim:{experiment_settings} : env:{envdet["skeleton"]}.{envdet["activity"]:6}  -  sim_iter:{it} - collision:{collision_simulation}  '
                f'-  car_pos{meta_data["car_pos"][0]:.2f},{meta_data["car_pos"][1]:.2f} | human:{meta_data["human_pos"]:.2f}/{meta_data["human_rot"]:.2f}  '
                f'- init:{env.ms2kmh(meta_data["car_speed"]):.2f} km/h   collision:{collision_speed:.2f} km/h')
        if fall:
            print(
                f'sim:{experiment_settings} : env:{envdet["skeleton"]}.{envdet["activity"]:6} -  '
                f'car_pos{meta_data["car_pos"][0]:.2f},{meta_data["car_pos"][1]:.2f} | human:{meta_data["human_pos"]:.2f}/{meta_data["human_rot"]:.2f}  '
                f'init:{env.ms2kmh(meta_data["car_speed"]):.2f} km/h   - no collision - agent falls alone')
        if len(logger.frame) == 0:
            print(
                f'sim:{experiment_settings} : env:{envdet["skeleton"]}.{envdet["activity"]:6} -  '
                f'car_pos{meta_data["car_pos"][0]:.2f},{meta_data["car_pos"][1]:.2f} | human:{meta_data["human_pos"]:.2f}/{meta_data["human_rot"]:.2f}  '
                f'init:{env.ms2kmh(meta_data["car_speed"]):.2f} km/h   - no collision***')
        if not activate_agent:
            processing_time2 = time.process_time()
            if len(video_data) > 0:
                # logger.add_video(experiment_settings * args.num_simulations_per_env + sim, video_data)
                logger.add_video(collision_simulation, video_data)
            meta_data_full = {"activity_type": envdet["activity"],
                              "car_type": envdet["car_type"],
                              "action_after_collision": envdet["after_collision"],
                              "humanoid_mode": envdet["skeleton"],
                              "last_iter": it,
                              "use_foot_forces": args.use_foot_forces,
                              "collision_iter": collision_it,
                              "collision_speed (km/h)": collision_speed,
                              "simluation_timestep": args.timestep,
                              "simulation_time": np.round(wrapper.env._data.time, 10),
                              "processing_time": np.round(processing_time2 - processing_time1, 4),
                              "seed": args.seed,
                              },
            logger.add_meta_data({"meta_data_env": meta_data,
                                  "meta_data_full": meta_data_full,
                                  })
            logger.record_simulation_meta(meta_data_full)
            logger.save_json_multiple_files()
        logger.clean_data()
        if video_data: video_data.clear()
        if done:
            collision_simulation += 1

        if (experiment_settings + 1) % args.num_simulations_per_env == 0:
            wrapper.env.stop()
            if renderer:
                del renderer
            del wrapper

    print("--- Test finished ---")


def str_to_bool(s):
    if s.lower() in ['true', 't', 'yes', 'y']:
        return True
    elif s.lower() in ['false', 'f', 'no', 'n']:
        return False
    elif s.lower() == 'none':
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_log", type=str,
                        help="Path to the new dataset generation")
    parser.add_argument("total_collisions_generated", type=int,
                        help="Total of collision simulation required to complete the experiment")
    parser.add_argument("num_simulations_per_env", type=int,
                        help="Times that produce the same Model (humanoid/car) simulation.")

    parser.add_argument("--agent_path", default="", type=str,
                        help="path to the pretrained agent")
    parser.add_argument("--render", default=None, type=str_to_bool,
                        help="None (default): nothing happens, True: run the Mujoco render, False: Video")

    parser.add_argument("--activity-type", default=["walk", "run"], nargs="+",
                        help="action types performed by the humanoid")
    parser.add_argument("--skeleton-type",
                        default=["Torque4Ages.4", "Torque4Ages.3", "Torque4Ages.2", "Torque4Ages.1"],
                        nargs="+",
                        help="Skeleton forces used by the humanoid. Torque, Torque4Ages.4, Torque4Ages.3, Torque4Ages.2, Torque4Ages.1 or Muscle")
    parser.add_argument("--car-type", default=["autocar", "prius"], nargs="+",
                        help="Car models used for the dataset generation")
    parser.add_argument("--after-collision-type", default=["zero", "move"], nargs="+",
                        help="Forces applied to the humanoid after the collision")

    ######################################
    ############## Init cases ############
    ######################################
    parser.add_argument("--init-car-speed", default=[5, 65], nargs="+",
                        help="init car speed set in the first frame of the environment")
    parser.add_argument("--init-car-turn", default=[-0.35, 0.35], nargs="+",
                        help="init car wheels turn set in the first frame of the environment")
    parser.add_argument("--init-car-position", default=[-5, 0], nargs="+",
                        help="init car position in Y axis set in the first frame of the environment")
    parser.add_argument("--init-car-position-shift", default=[-0.4, 0.4], nargs="+",
                        help="init car position in X in the first frame of the environment")
    parser.add_argument("--init-human-position", default=[-2, -0], nargs="+",
                        help="init humanoid position in the first frame of the environment")
    parser.add_argument("--init-human-orientation", default=[-45, 45], nargs="+",
                        help="init humanoid orientation/rotation in the first frame of the environment")
    parser.add_argument("--flip-human", default=True,
                        help="if humanoid performs the mirror case. Also rotation changes to the oposite angles")

    parser.add_argument("--use_foot_forces", default=False, choices=[True, False],
                        help="Use foot forces for the locomujoco humanoid")
    parser.add_argument("--timestep", default=0.5e-3, type=float,
                        help="simulation timestep. Consider always this no higher than 1ms")
    parser.add_argument("--n_substeps", default=1, type=int,
                        help="Times to mujoco repeat the last action inside the environment before to go to the next iteration")
    parser.add_argument("--cameras",
                        default=["track_car_top", "track_car_side", "track_car_back", "track_driver", "track_human"],
                        type=list, help="mujoco cameras to record the videos")
    parser.add_argument("--init-collision", type=int,
                        help="init counter to blend with previous joints")
    parser.add_argument("--seed", default=5, type=int,
                        help="Experiment seed")
    args = parser.parse_args()
    main(args)
