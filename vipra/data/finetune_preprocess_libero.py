import argparse
import json

import pandas as pd


def assign_bin(data_point, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= data_point < bins[i + 1]:
            return i
    if data_point >= bins[len(bins) - 1]:
        return len(bins) - 2
    if data_point <= bins[0]:
        return 0
    return None  # For data points outside the bin range


def main(args):
    input_jsonl = []
    output_jsonl = []

    # read json file not jsonl
    with open(args.input_path, "r") as infile:
        for line in infile:
            input_jsonl.append(json.loads(line))

    # Extracting action lists
    total_list = [[] for _ in range(len(input_jsonl[0]["raw_action"]))]

    for instance in input_jsonl:
        action = instance["raw_action"]
        action = [float(action[i]) for i in range(len(action))]
        assert len(action) == len(total_list)
        for i in range(len(action)):
            total_list[i].append(action[i])

    # import ipdb; ipdb.set_trace()
    total_bin = []

    for index, individual_list in enumerate(total_list):
        action_series = pd.Series(individual_list)
        discretized_data, bins = pd.qcut(
            action_series, args.discretize_bins, labels=False, retbins=True, duplicates="drop"
        )
        total_bin.append(bins)
        print(f"bin {index}:", bins)
        print(f"Number of bins: {len(bins)}")

    for index, instance in enumerate(input_jsonl):
        out_data = {}
        out_data["id"] = args.task_name + "/" + instance["id"]
        out_data["image"] = args.task_name + "_modified/images/" + instance["image"]
        action = instance["raw_action"]
        out_data["raw_action"] = action
        out_data["proprio"] = instance["proprio"]

        action = [float(action[i]) for i in range(len(action))]
        action_quant = [assign_bin(action[i], total_bin[i]) for i in range(len(action))]
        # binarize gripper action
        action_quant[-1] = 1 if action[-1] > 0 else 0  # [-1, 1] -> [0, 1]
        out_data["action"] = action_quant
        instruction = instance["instruction"]
        out_data["instruction"] = instruction

        output_jsonl.append(out_data)

    # Save each category to its own JSONL file
    with open(args.output_filename, "w") as outfile:
        for step in output_jsonl:
            outfile.write(json.dumps(step) + "\n")

    total_bin[-1] = [-1.0, 0.0, 1.0]  # Gripper action is binarized, so we set the bins to [-1, 0, 1]
    df = pd.DataFrame(total_bin)
    df.to_csv(args.csv_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON data and save processed actions and bins.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_filename", type=str, required=True, help="Path to save the output JSONL file prefix.")
    parser.add_argument("--csv_filename", type=str, required=True, help="Path to save the bins as a CSV file.")
    parser.add_argument("--discretize_bins", type=int, default=2047, help="Number of bins to discretize the actions.")
    parser.add_argument("--task_name", type=str, help="Libero task name")
    args = parser.parse_args()
    main(args)
    """
    python data/finetune_preprocess_libero.py --input_path "/fsx2/shared/sroutray/LIBERO/libero_10_modified/libero_10_raw.jsonl" --output_filename "/fsx2/shared/sroutray/LIBERO/libero_10_modified/libero_10_quant_v2.jsonl" --csv_filename "/fsx2/shared/sroutray/LIBERO/libero_10_modified/quant_bins_v2.csv" --task_name "libero_10"
    """
