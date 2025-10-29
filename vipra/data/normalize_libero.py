import json
from pathlib import Path

import numpy as np


def main(args):
    # Step 1: Compute mean/std from raw.jsonl
    all_actions = []
    all_proprio = []
    with open(args.raw_jsonl, "r") as f:
        for line in f:
            d = json.loads(line)
            all_actions.append(d["raw_action"])
            all_proprio.append(d["proprio"])

    actions_np = np.array(all_actions)
    mean = actions_np.mean(axis=0)
    std = actions_np.std(axis=0) + 1e-6

    proprio_np = np.array(all_proprio)
    proprio_mean = proprio_np.mean(axis=0)
    proprio_std = proprio_np.std(axis=0) + 1e-6

    # Do not normalize gripper (last dimension)
    gripper_idx = -1
    mean[gripper_idx] = 0
    std[gripper_idx] = 1

    # Save normalization stats
    norm_stats = {"mean": mean.tolist(), "std": std.tolist()}
    with open(args.action_stats_json, "w") as f:
        json.dump(norm_stats, f, indent=2)

    proprio_stats = {"mean": proprio_mean.tolist(), "std": proprio_std.tolist()}
    with open(args.proprio_stats_json, "w") as f:
        json.dump(proprio_stats, f, indent=2)

    # Step 2: Normalize action_cont in dynamics14
    with open(args.dynamics_jsonl, "r") as f:
        dynamics_data = [json.loads(line) for line in f]

    for d in dynamics_data:
        raw = np.array(d["action_cont"])
        raw = raw.reshape(-1, len(mean))  # shape (T, D)
        normed = (raw - mean) / std  # broadcast normalization
        d["action_cont"] = normed.flatten().tolist()
        # Also normalize proprio
        proprio = np.array(d["proprio"])
        proprio = proprio.reshape(-1, len(proprio_mean))
        normed_proprio = (proprio - proprio_mean) / proprio_std
        d["proprio"] = normed_proprio.flatten().tolist()

    # Save output
    with open(args.output_jsonl, "w") as f:
        for d in dynamics_data:
            f.write(json.dumps(d) + "\n")

    print(f"[✓] Normalized actions saved to {args.output_jsonl}")
    print(f"[✓] Normalization stats saved to {args.action_stats_json}")
    print(f"[✓] Proprio normalization stats saved to {args.proprio_stats_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize actions and proprioceptive states for stable training.")
    parser.add_argument(
        "--raw_jsonl", type=str, required=True, help="Path to raw JSONL file for computing normalization stats"
    )
    parser.add_argument("--dynamics_jsonl", type=str, required=True, help="Path to dynamics JSONL file to normalize")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save normalized output JSONL")
    parser.add_argument(
        "--action_stats_json", type=str, required=True, help="Path to save action normalization stats JSON"
    )
    parser.add_argument(
        "--proprio_stats_json", type=str, required=True, help="Path to save proprioceptive normalization stats JSON"
    )

    args = parser.parse_args()
    main(args)
