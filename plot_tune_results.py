from tfm_utils.plot_tune_utils import parallel_coord_plot, plot_hyperparameters, plot_cost_trajectories
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--title', type=str, required=True)
parser.add_argument('--keys', type=str, nargs='*', required=True)
parser.add_argument('--csv_paths', type=str, nargs='*', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--window_size', type=int, required=False, default=50)
parser.add_argument('--configs', type=int, required=False, default=100)
args = parser.parse_args()

data_dict = dict(zip(args.keys, args.csv_paths))

# Cost Trajectories
for key, path in zip(args.keys, args.csv_paths):
    plot_cost_trajectories({key:path}, path=os.path.join(args.out_dir, "time_vs_cost",  args.title, f"{key}.png"), window_size=args.window_size)
plot_cost_trajectories(data_dict, path=os.path.join(args.out_dir, "time_vs_cost", args.title, "all.png"), window_size=args.window_size)

# HPs vs Cost
for key, path in zip(args.keys, args.csv_paths):
    plot_hyperparameters({key:path}, suptitle=f"{key}_{args.title}", path=os.path.join(args.out_dir, "hp_vs_cost", args.title, f"{key}.png"))

# Parallel Coord Plots
for key, path in zip(args.keys, args.csv_paths):
    parallel_coord_plot(
        csv_file=path,
        suptitle=f"{key}_{args.title}_{args.configs}configs",
        path=os.path.join(args.out_dir, "parallel_coord_plots", args.title, f"{key}.png"),
        configs=args.configs,
        )