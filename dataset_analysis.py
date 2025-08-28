from pathlib import Path

import numpy as np

import pedestrian_impact as env


def main(args):
    if args.idx is not None:
        data = env.data_utils.get_mlarki_data_from_json(path=args.experiment_path,
                                                        idx=args.idx,
                                                        data_type="joint",
                                                        joint_type="qpos")
        mlarki_data = env.data_utils.convert_mlarki_qs_to_body(data["meta"], data["data"], attention=True)

        data = env.data_utils.get_numpy_data_from_json(path=args.experiment_path,
                                                       idx=args.idx,
                                                       data_type="joint",
                                                       joint_type="qpos")
        np_xpos = env.data_utils.convert_numpy_qs_to_body(data["meta"], data["data"])
        skeleton = env.data_utils.get_skeleton("attention")
        print("done! copy this to jupyter for mlarki visualization =)")

    ###############################################################
    ###############################################################
    ###############################################################
    if args.save_numpy:
        path = Path(args.experiment_path)
        files = list(path.joinpath("files").glob("*.json"))
        files.sort()
        files = np.array(files)
        data = None
        path.joinpath("numpy").mkdir(parents=True, exist_ok=True)
        # Get all similar environments to read this as a batch.
        dataset = env.data_utils.load_json(path.joinpath("dataset_summary.json"))
        meta_collection = []
        for key, info in dataset.items():
            meta_collection.append(f'{info[0]["car_type"]}_{info[0]["humanoid_mode"]}')
        meta_collection = np.array(meta_collection)
        unique_meta_env = np.unique(meta_collection)
        summed_idxs = 0
        for unique_env in unique_meta_env:
            idxs = np.where(meta_collection == unique_env)[0]
            first_env = True
            for i, file in enumerate(files[idxs]):
                out_file = file.parent.parent.joinpath("numpy").joinpath(file.stem + ".npy")
                if (not out_file.exists()) or out_file.stat().st_size == 0:
                    data = env.data_utils.get_numpy_data_from_json(path=file,
                                                                   idx=i,  # ignore this one
                                                                   data_type="joint",
                                                                   joint_type="qpos")
                    # xpos = env.data_utils.convert_numpy_qs_to_body(data["meta"], data["data"])
                    if first_env:
                        _, m, d = env.data_utils._make_env(data["meta"].data.meta_data_full[0])
                        first_env = False
                    xpos = env.data_utils.efficient_convert_numpy_qs_to_body(data["meta"], data["data"], m, d)
                    np.save(out_file, xpos)
                print(f'{(summed_idxs + i) / len(files) * 100:.2f}', end='\r', flush=True)
            summed_idxs += len(idxs)
        if not files[0].parent.parent.joinpath("joints.npy").exists():
            if data is None:
                data = env.data_utils.get_numpy_data_from_json(path=files[0],
                                                               idx=0,
                                                               data_type="joint",
                                                               joint_type="qpos")
            np.save(files[0].parent.parent.joinpath("joints"), data["joints"])
        if not files[0].parent.parent.joinpath("bodys.npy").exists():
            if data is None:
                data = env.data_utils.get_numpy_data_from_json(path=files[0],
                                                               idx=0,
                                                               data_type="joint",
                                                               joint_type="qpos")
            np.save(files[0].parent.parent.joinpath("body"), env.data_utils.get_body_names(data["meta"]))
        print("\nAll numpy files have been converted...")

    if args.merge:
        path = Path(args.experiment_path)
        files = list(path.rglob("dataset_summary.json"))
        output_path = path.joinpath("merged_dataset")
        output_path.mkdir(parents=True, exist_ok=True)
        if output_path.joinpath("dataset_summary.json") in files:
            files.remove(output_path.joinpath("dataset_summary.json"))
        files = np.array(files)
        datasets = {}
        for file in files:
            datasets.update(env.data_utils.load_json(file))
        datasets = dict(sorted(datasets.items()))
        env.data_utils.save_json(output_path.joinpath("dataset_summary.json"), datasets)
        del datasets

        for data_type, ext in zip(["files", "numpy", "videos"], ["json", "npy", ".mp4"]):
            print(f'Processing: {data_type}')
            output_path.joinpath(data_type).mkdir(parents=True, exist_ok=True)

            json_collection = []
            for file in files:
                jsons = list(file.parent.joinpath(data_type).glob(f'*.{ext}'))
                json_collection.extend(jsons)
            if len(json_collection) != len(set(json_collection)):
                raise AssertionError(
                    "Some json files are repeated (e.g. sim0000 sim0000 sim0001). This process will be randomly removed in the merge process")

            for i, json_file in enumerate(json_collection):
                if not output_path.joinpath(data_type, json_file.name).exists():
                    output_path.joinpath(data_type, json_file.name).write_bytes(json_file.read_bytes())
                print(f'{i / len(json_collection) * 100:.2f}', end='\r', flush=True)
        del json_collection
        bodys = json_file.parent.parent.joinpath("body.npy")
        output_path.joinpath("body.npy").write_bytes(bodys.read_bytes())
        joints = json_file.parent.parent.joinpath("joints.npy")
        output_path.joinpath("joints.npy").write_bytes(joints.read_bytes())

    if args.split:
        path = Path(args.experiment_path)
        files = list(path.joinpath("numpy").glob("*.npy"))
        np.random.seed(0)
        idxs = np.arange(len(files))
        files = np.array(files)[idxs]
        if len(args.split) != 3:
            raise AssertionError("split must have exactly 3 sets. Training, Validation and Testing")
        prev_split = 0
        for split, group in zip(args.split, ["train.txt", "val.txt", "test.txt"]):
            print(int(prev_split / 100 * len(files)), int((prev_split + int(split)) / 100 * len(files)))
            curr_files = files[int(prev_split / 100 * len(files)):int((prev_split + int(split)) / 100 * len(files))]
            file = open(path.joinpath(group), "w")
            for i, npy in enumerate(curr_files):
                file.write(f'{str(npy)}\n')
            file.close()
            prev_split += int(split)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", help="path to dataset")
    parser.add_argument("--idx", type=int, help="index of the simulation")
    parser.add_argument("--save-numpy", action="store_true", help="save into a numpy folder")
    parser.add_argument("--merge", action="store_true", help="save into a numpy folder")
    parser.add_argument("--split", nargs="+", help="save into a numpy folder")
    args = parser.parse_args()
    main(args)
