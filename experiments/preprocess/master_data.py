import cv2
import pandas as pd
import os
import numpy as np
import tqdm
import json

def join_helmets_contact(game_play, labels, meta, view="Sideline", fps=59.94):
    """
    Joins helmets and labels for a given game_play. Results can be used for visualizing labels.
    Returns a dataframe with the joint dataframe, duplicating rows if multiple contacts occur.
    """
    gp_labels = labels.query("game_play == @game_play").copy()
    start_time = meta.query("game_play == @game_play and view == @view")[
        "start_time"
    ].values.astype('datetime64[s]')[0]
    gp_labels["frame"] = (gp_labels["datetime"].values.astype('datetime64[ms]') - start_time) * fps
    gp_labels["frame"] = gp_labels["frame"].dt.seconds
    return gp_labels

base_dir = "../../input/nfl-player-contact-detection"

df_labels = pd.read_csv(f"{base_dir}/train_labels.csv", parse_dates=["datetime"])
df_meta = pd.read_csv(f"{base_dir}/train_video_metadata.csv", parse_dates=["start_time", "end_time", "snap_time"])
df_tracking = pd.read_csv(f"{base_dir}/train_player_tracking.csv", parse_dates=["datetime"])

df_labels["nfl_player_id_1"] = df_labels["nfl_player_id_1"].astype(str)
df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)
game_plays = df_labels["game_play"].drop_duplicates().values

gps = []
for game_play in tqdm.tqdm(game_plays):
    for view in ["Endzone"]:  # Endzone, Sidelineは時間同期されている
        gp = join_helmets_contact(game_play, df_labels, df_meta, view=view)
        gps.append(gp)

gps = pd.concat(gps)

gps = pd.merge(
    gps,
    df_tracking.rename(columns={col: f"{col}_1" if col not in ["game_play", "datetime"] else col for col in df_tracking.columns}),
    how="left"
)
gps = pd.merge(
    gps,
    df_tracking.rename(columns={col: f"{col}_2" if col not in ["game_play", "datetime"] else col for col in df_tracking.columns}),
    how="left"
)

gps["distance"] = np.sqrt(
    (gps["x_position_1"].values - gps["x_position_2"]) ** 2 + (gps["y_position_1"].values - gps["y_position_2"]) ** 2)

output_dir = "../../output/preprocess/master_data/"
os.makedirs(output_dir, exist_ok=True)
gps.reset_index(drop=True).to_feather(f"{output_dir}/gps.feather")