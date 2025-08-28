import pedestrian_impact as env


def main(args):
    # experiment_path = "logs/20240330_1253-id1994/"
    viewer = env.Visualizer(args.experiment_path, args.idx)
    viewer.lauch(show_left_ui=False, show_right_ui=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", help="path to dataset")
    parser.add_argument("idx", type=int, help="index of the simulation")
    args = parser.parse_args()
    main(args)
