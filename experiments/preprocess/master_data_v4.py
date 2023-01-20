import cv2
import pandas as pd
import os
import numpy as np
import tqdm
import json


def join_helmets_contact(game_play, labels, helmets, meta, view="Sideline", fps=59.94):
    """
    Joins helmets and labels for a given game_play. Results can be used for visualizing labels.
    Returns a dataframe with the joint dataframe, duplicating rows if multiple contacts occur.
    """
    gp_labs = labels.query("game_play == @game_play").copy()
    gp_helms = helmets.query("game_play == @game_play and view == @view").copy()

    start_time = meta.query("game_play == @game_play and view == @view")[
        "start_time"
    ].values[0]

    gp_helms["datetime"] = (
        pd.to_timedelta(gp_helms["frame"] * (1 / fps), unit="s") + start_time
    )
    gp_helms["datetime"] = pd.to_datetime(gp_helms["datetime"], utc=True)
    gp_helms["datetime_ngs"] = (
        pd.DatetimeIndex(gp_helms["datetime"] + pd.to_timedelta(50, "ms"))
            .floor("100ms")
            .values
    )
    gp_helms["datetime_ngs"] = pd.to_datetime(gp_helms["datetime_ngs"], utc=True)
    gp_labs["datetime_ngs"] = pd.to_datetime(gp_labs["datetime"], utc=True)

    # TODO: helmetも player_1, player_2 ごとに抽出
    helmet_columns = [
        "frame", "datetime_ngs", "datetime", "view", "nfl_player_id", "left", "width", "top", "height",
        "x", "y", "team"
    ]
    gp = pd.merge(
        gp_labs[["contact_id", "game_play", "step", "nfl_player_id_1", "nfl_player_id_2", "datetime_ngs", "contact"]],
        gp_helms[helmet_columns].rename(
            columns={col: f"{col}_1" if col not in ["frame", "datetime_ngs", "datetime", "view"] else col for col in helmet_columns}
        ),
        how="left",
    )
    gp = pd.merge(
        gp,
        gp_helms[helmet_columns].rename(
            columns={col: f"{col}_2" if col not in ["frame", "datetime_ngs", "datetime", "view"] else col for col in helmet_columns}
        ),
        how="left",
    )
    gp["diff"] = np.abs(gp["datetime"] - gp["datetime_ngs"])
    gp["rank"] = gp.groupby(["contact_id", "nfl_player_id_1", "nfl_player_id_2"])["diff"].rank(method="first")
    gp = gp[(gp["rank"] == 1) | (gp["rank"].isnull())]
    gp["datetime"] = gp["datetime_ngs"]

    gp_ret = []
    for key, gp_ in gp.groupby(["datetime_ngs"]):
        frame = np.round(gp_["frame"].mean())
        gp_["frame"] = gp_["frame"].fillna(frame)
        gp_ret.append(gp_)
    gp = pd.concat(gp_ret)
    gp["frame"] = gp["frame"].fillna(0).astype(int)
    gp = gp.drop(["datetime_ngs", "diff", "rank"], axis=1)
    gp = gp[gp["frame"].notnull()]

    gp["game_play"] = gp["game_play"].fillna(game_play)
    gp["view"] = gp["view"].fillna(view)
    return gp


base_dir = "../../input/nfl-player-contact-detection"

df_labels = pd.read_csv(f"{base_dir}/train_labels.csv", parse_dates=["datetime"])
df_meta = pd.read_csv(f"{base_dir}/train_video_metadata.csv", parse_dates=["start_time", "end_time", "snap_time"])
df_tracking = pd.read_csv(f"{base_dir}/train_player_tracking.csv", parse_dates=["datetime"])
df_helmets = pd.read_csv(f"{base_dir}/train_baseline_helmets.csv")
df_tracking["datetime"] = pd.to_datetime(df_tracking["datetime"], utc=True)

df_labels["nfl_player_id_1"] = df_labels["nfl_player_id_1"].astype(str)
df_helmets["nfl_player_id"] = df_helmets["nfl_player_id"].astype(str)
df_helmets["x"] = df_helmets["left"] + df_helmets["width"] / 2
df_helmets["y"] = df_helmets["top"] + df_helmets["height"] / 2
df_helmets["team"] = [x[0] for x in df_helmets["player_label"].values]
df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)
game_plays = df_labels["game_play"].drop_duplicates().values

df_helmets_concat = []
for key, df_ in tqdm.tqdm(df_helmets.groupby(["game_play", "view", "nfl_player_id", "team"])):
    frame_min = df_["frame"].min()
    frame_max = df_["frame"].max()
    game_play = key[0]
    view = key[1]
    nfl_player_id = key[2]
    team = key[3]

    df_ = pd.merge(
        pd.DataFrame({"frame": np.arange(frame_min, frame_max + 1)}).astype(int),
        df_,
        how="left"
    )
    df_["game_play"] = game_play
    df_["nfl_player_id"] = nfl_player_id
    df_["view"] = view
    df_["team"] = team
    df_helmets_concat.append(df_)

df_helmets = pd.concat(df_helmets_concat)

gps = []

for key, w_df_helms in tqdm.tqdm(df_helmets.groupby(["game_play", "view"])):
    try:
        game_play = key[0]
        view = key[1]
        gp = join_helmets_contact(game_play, df_labels, w_df_helms, df_meta, view=view)
        gps.append(gp)
    except Exception as e:
        print(e)

gps = pd.concat(gps)

tracking_columns = [
    "game_play", "game_key", "nfl_player_id", "datetime", "step", "position", "x_position", "y_position",
    "speed", "distance", "direction", "orientation", "acceleration", "sa"
]
key_columns = ["game_play", "datetime", "game_key", "step"]
gps = pd.merge(
    gps,
    df_tracking[tracking_columns].rename(columns={col: f"{col}_1" if col not in key_columns else col for col in tracking_columns}),
    how="left"
)
gps = pd.merge(
    gps,
    df_tracking[tracking_columns].rename(columns={col: f"{col}_2" if col not in key_columns else col for col in tracking_columns}),
    how="outer"
)

gps["distance"] = np.sqrt(
    (gps["x_position_1"].values - gps["x_position_2"]) ** 2 + (gps["y_position_1"].values - gps["y_position_2"]) ** 2
)

print(gps.isnull().sum() / len(gps))
gps["frame"] = gps["frame"].fillna(99999).astype(int)
output_dir = "../../output/preprocess/master_data_v4/"
os.makedirs(output_dir, exist_ok=True)
gps.reset_index(drop=True).to_feather(f"{output_dir}/gps.feather")