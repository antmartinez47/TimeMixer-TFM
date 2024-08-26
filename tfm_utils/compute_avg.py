import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', type=str, required=True)
parser.add_argument('--itr', type=int, required=True)
args = parser.parse_args()

df_list = []
for i in range(args.itr):
    df_list.append(pd.read_csv(os.path.join(args.dirpath, f"exp{i}", "best_metrics.csv")))

avg_dir = f"checkpoints/baseline/TimeMixer/{dir}/avg"
avg_dir = os.path.join(args.dirpath, "avgexp")
if not os.path.isdir(avg_dir):
    os.makedirs(avg_dir)

avg_df = pd.concat(df_list, axis=0).mean().to_frame().T.drop("Unnamed: 0", axis=1)
avg_df.to_csv(os.path.join(avg_dir, "avg_metrics.csv"))