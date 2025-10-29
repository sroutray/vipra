import csv
import json
import os
from pathlib import Path


def create_dynamics_jsonl(jsonl_path, data_root, horiz, zero_action_idx, action_type="abs-joint"):
    jsonl_data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            jsonl_data.append(json.loads(line))

    # Build mapping from trajectory (task/episode_id) to list of frame dicts
    # Example id: 'libero_10/ep00000/step0001'
    traj_dict = {}
    for data in jsonl_data:
        parts = data["id"].split("/")
        tid = "/".join(parts[:-1])  # removes last /xxxx
        traj_dict.setdefault(tid, []).append(data)

    # Sort timesteps within each trajectory
    for traj_id in traj_dict:
        traj_dict[traj_id].sort(key=lambda x: x["id"])

    dynamics_jsonl = []

    for traj_id, traj_data in traj_dict.items():
        for id in range(len(traj_data) - 1):
            data = dict()
            data["id"] = traj_data[id]["id"]
            data["instruction"] = traj_data[id]["instruction"]

            # history frames
            id_h1 = max(id - 2, 0)
            data["image"] = [traj_data[id_h1]["image"], traj_data[id]["image"]]
            data["action"] = []
            data["action_cont"] = []

            # history proprio
            # id_ph6, ..., id_ph1, id
            # Repeat if not enough history
            data["proprio"] = []
            for h in range(6, -1, -1):
                id_ph = max(id - h, 0)
                data["proprio"].extend(traj_data[id_ph]["proprio"])

            for future_id in range(horiz):
                id_f = id + future_id
                if id_f < len(traj_data):
                    data["action"].extend(traj_data[id_f]["action"])
                    data["action_cont"].extend(traj_data[id_f]["raw_action"])
                else:
                    if action_type == "abs-joint":
                        data["action"].extend(traj_data[-1]["action"])
                        data["action_cont"].extend(traj_data[-1]["raw_action"])
                    elif action_type == "delta-eef":
                        action = zero_action_idx[:]
                        action[-1] = traj_data[-1]["action"][-1]
                        action_cont = [0.0] * len(action)
                        action_cont[-1] = traj_data[-1]["raw_action"][-1]
                        data["action"].extend(action)
                        data["action_cont"].extend(action_cont)
                    else:
                        raise ValueError("Invalid action type")

            term_id = min(id + horiz, len(traj_data) - 1)
            data["latent_state"] = [traj_data[term_id]["image"]]
            data["fields_ls_qa"] = "[instruction],[vision],latent_state,action"
            data["fields_qa"] = "[instruction],[vision],action"
            data["fields_ca"] = "[instruction],[vision],action_cont"
            data["fields_ph_ca"] = "[instruction],[vision],[proprio],action_cont"
            data["raw_action"] = traj_data[id]["raw_action"]

            dynamics_jsonl.append(data)

    # Save output
    parent_dir = Path(data_root).resolve()
    output_path = parent_dir / f"libero_10_dynamics{horiz}_v2.jsonl"
    with open(output_path, "w") as f:
        for d in dynamics_jsonl:
            f.write(json.dumps(d) + "\n")
    print(f"[✓] Saved {len(dynamics_jsonl)} dynamics records to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dynamics JSONL with action sequences and temporal context.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to quantized input JSONL file")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for the dataset")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to quantization bins CSV file")
    parser.add_argument("--horizon", type=int, default=14, help="Action sequence horizon length")
    parser.add_argument(
        "--action_type",
        type=str,
        choices=["abs-joint", "delta-eef"],
        default="delta-eef",
        help="Action type for padding",
    )
    parser.add_argument("--task_name", type=str, default="libero_10", help="Task name for output file")

    args = parser.parse_args()

    action_scale_list = []
    with open(args.csv_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            action_scale_list.append([float(val) for val in row if val.strip()])

    # Find "zero" index per action dimension (used in delta-eef padding)
    zero_action_idx = []
    for values in action_scale_list:
        min_diff, min_diff_idx = float("inf"), 0
        for i in range(len(values) - 1):
            avg = (values[i] + values[i + 1]) / 2
            diff = abs(avg)
            if diff < min_diff:
                min_diff = diff
                min_diff_idx = i
        zero_action_idx.append(min_diff_idx)

    # Update the function to use task_name for output filename
    def create_dynamics_jsonl_with_args(jsonl_path, data_root, horiz, zero_action_idx, action_type, task_name):
        jsonl_data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                jsonl_data.append(json.loads(line))

        # Build mapping from trajectory (task/episode_id) to list of frame dicts
        traj_dict = {}
        for data in jsonl_data:
            parts = data["id"].split("/")
            tid = "/".join(parts[:-1])  # removes last /xxxx
            traj_dict.setdefault(tid, []).append(data)

        # Sort timesteps within each trajectory
        for traj_id in traj_dict:
            traj_dict[traj_id].sort(key=lambda x: x["id"])

        dynamics_jsonl = []

        for traj_id, traj_data in traj_dict.items():
            for id in range(len(traj_data) - 1):
                data = dict()
                data["id"] = traj_data[id]["id"]
                data["instruction"] = traj_data[id]["instruction"]

                # history frames
                id_h1 = max(id - 2, 0)
                data["image"] = [traj_data[id_h1]["image"], traj_data[id]["image"]]
                data["action"] = []
                data["action_cont"] = []

                # history proprio
                data["proprio"] = []
                for h in range(6, -1, -1):
                    id_ph = max(id - h, 0)
                    data["proprio"].extend(traj_data[id_ph]["proprio"])

                for future_id in range(horiz):
                    id_f = id + future_id
                    if id_f < len(traj_data):
                        data["action"].extend(traj_data[id_f]["action"])
                        data["action_cont"].extend(traj_data[id_f]["raw_action"])
                    else:
                        if action_type == "abs-joint":
                            data["action"].extend(traj_data[-1]["action"])
                            data["action_cont"].extend(traj_data[-1]["raw_action"])
                        elif action_type == "delta-eef":
                            action = zero_action_idx[:]
                            action[-1] = traj_data[-1]["action"][-1]
                            action_cont = [0.0] * len(action)
                            action_cont[-1] = traj_data[-1]["raw_action"][-1]
                            data["action"].extend(action)
                            data["action_cont"].extend(action_cont)
                        else:
                            raise ValueError("Invalid action type")

                term_id = min(id + horiz, len(traj_data) - 1)
                data["latent_state"] = [traj_data[term_id]["image"]]
                data["fields_ls_qa"] = "[instruction],[vision],latent_state,action"
                data["fields_qa"] = "[instruction],[vision],action"
                data["fields_ca"] = "[instruction],[vision],action_cont"
                data["fields_ph_ca"] = "[instruction],[vision],[proprio],action_cont"
                data["raw_action"] = traj_data[id]["raw_action"]

                dynamics_jsonl.append(data)

        # Save output
        parent_dir = Path(data_root).resolve()
        output_path = parent_dir / f"{task_name}_dynamics{horiz}_v2.jsonl"
        with open(output_path, "w") as f:
            for d in dynamics_jsonl:
                f.write(json.dumps(d) + "\n")
        print(f"[✓] Saved {len(dynamics_jsonl)} dynamics records to {output_path}")

    create_dynamics_jsonl_with_args(
        args.input_jsonl, args.data_root, args.horizon, zero_action_idx, args.action_type, args.task_name
    )
