import cv2
import pandas as pd
import os
import numpy as np
import tqdm


CONTACT = (0, 0, 0)
AWAY = (0, 0, 255)
HOME = (128, 255, 0)
G = (255, 255, 255)
output_size = (128, 96)
output_dir = f"../../../work/images_{output_size[0]}x{output_size[1]}_v19"
traintest = "train"

bbox_left_ratio = 4.5
bbox_right_ratio = 4.5
bbox_top_ratio = 4.5
bbox_down_ratio = 2.25

def load_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    imgs = []
    for _ in range(frame_count):
        it_worked, img = vidcap.read()
        imgs.append(img)
    imgs = np.stack(imgs)
    return imgs

def join_helmets_contact(game_play, labels, helmets, meta, view="Sideline", fps=59.94):
    """
    Joins helmets and labels for a given game_play. Results can be used for visualizing labels.
    Returns a dataframe with the joint dataframe, duplicating rows if multiple contacts occur.
    """
    gp_labs = labels.query("game_play == @game_play").copy()
    gp_helms = helmets.query("game_play == @game_play").copy()

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

    keys = ["frame", "game_play", "datetime_ngs", "view"]
    gp = pd.merge(
        gp_labs,
        gp_helms.drop("datetime", axis=1).rename(columns={col: f"{col}_1" for col in gp_helms.columns if col not in keys}),
        how="left",
    )
    gp = pd.merge(
        gp,
        gp_helms.drop("datetime", axis=1).rename(columns={col: f"{col}_2" for col in gp_helms.columns if col not in keys}),
        how="left",
    )
    gp["nfl_player_id_2"] = ["G" if "G" in ary[1] else ary[0] for ary in gp[["nfl_player_id_2", "contact_id"]].values]
    return gp


def main():
    base_dir = "../../input/nfl-player-contact-detection"

    df_helmets = pd.read_csv(f"{base_dir}/train_baseline_helmets.csv")
    df_helmets["x"] = df_helmets["left"] + df_helmets["width"] / 2
    df_helmets["y"] = df_helmets["top"] + df_helmets["height"] / 2
    df_helmets["team"] = [x[0] for x in df_helmets["player_label"].values]

    df_labels = pd.read_csv(f"{base_dir}/train_labels.csv", parse_dates=["datetime"])
    df_meta = pd.read_csv(f"{base_dir}/train_video_metadata.csv", parse_dates=["start_time", "end_time", "snap_time"])
    df_labels["nfl_player_id_1"] = df_labels["nfl_player_id_1"].astype(str)
    df_helmets["nfl_player_id"] = df_helmets["nfl_player_id"].astype(str)

    print(df_labels["contact"].sum(), len(df_labels))

    df_tracking = pd.read_feather("../../output/preprocess/master_data_v4/gps.feather")
    df_tracking["distance"] = [0 if ary[0] == "G" else ary[1] for ary in df_tracking[["nfl_player_id_2", "distance"]].values]
    df_dist = df_tracking.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])["distance"].min().reset_index()
    df_labels = pd.merge(df_labels, df_dist.query("distance < 1.5")[["game_play", "nfl_player_id_1", "nfl_player_id_2"]])

    print(df_labels["contact"].sum(), len(df_labels))

    df_helmets_ = df_helmets[[
        "game_play", "nfl_player_id", "x", "y", "left", "top", "width", "height", "frame", "view", "team"
    ]]
    df_helmets_concat = []
    for key, df_ in tqdm.tqdm(df_helmets_.groupby(["game_play", "view", "nfl_player_id", "team"])):
        frame_min = df_["frame"].min()
        frame_max = df_["frame"].max()
        game_play = key[0]
        view = key[1]
        nfl_player_id = key[2]
        team = key[3]

        df_ = pd.merge(
            pd.DataFrame({"frame": np.arange(frame_min, frame_max+1)}),
            df_,
            how="left"
        )
        df_["game_play"] = game_play
        df_["nfl_player_id"] = nfl_player_id
        df_["view"] = view
        df_["team"] = team
        # for col in ["x", "y", "left", "top", "width", "height"]:
        #     df_[col] = df_[col].interpolate(limit=5, limit_area="inside")
        df_helmets_concat.append(df_)

    df_helmets = pd.concat(df_helmets_concat)

    game_plays = df_labels["game_play"].drop_duplicates().values

    for i, game_play in enumerate(tqdm.tqdm(game_plays)):
        gp = join_helmets_contact(game_play, df_labels, df_helmets, df_meta)
        for col in ["x", "y", "width", "height"]:
            gp[col] = gp[[f"{col}_1", f"{col}_2"]].mean(axis=1)
        gp["bbox_size"] = gp[["width", "height"]].mean(axis=1)
        for view in ["Endzone", "Sideline"]:
            gp_ = gp[gp["view"] == view]

            data_dict = {}
            for key, w_df in gp_.groupby(["nfl_player_id_1", "nfl_player_id_2"]):
                w_df = w_df.set_index("frame")
                data_dict[key] = w_df

            bbox_dict = {}
            for key, w_df in gp_.drop_duplicates(["frame", "nfl_player_id_1"]).groupby("frame"):
                bbox_dict[key] = w_df[["left_1", "width_1", "top_1", "height_1", "team_1"]].dropna().values

            frames = gp_["frame"].drop_duplicates().values
            video_path = f"{base_dir}/{traintest}/{game_play}_{view}.mp4"
            # imgs = load_video(video_path)
            vidcap = cv2.VideoCapture(video_path)
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame in range(frame_count):
                it_worked, img_ = vidcap.read()

                if frame not in frames:
                    continue

                for bbox in bbox_dict[frame]:
                    box_left = int(bbox[0])
                    box_right = int(bbox[0] + bbox[1])
                    box_top = int(bbox[2])
                    box_down = int(bbox[2] + bbox[3])
                    if bbox[4] == "V":
                        color = AWAY
                    else:
                        color = HOME
                    cv2.rectangle(
                        img_,
                        (box_left, box_top),
                        (box_right, box_down),
                        color,
                        thickness=2,
                    )
                for key, w_df in data_dict.items():
                    img = img_.copy()

                    if not frame in w_df.index:
                        continue
                    series = w_df.loc[frame]
                    if np.isnan(series["x"]):
                        continue
                    left = int(series["x"] - series["bbox_size"] * bbox_left_ratio)
                    right = int(series["x"] + series["bbox_size"] * bbox_right_ratio)
                    top = int(series["y"] + series["bbox_size"] * bbox_top_ratio)
                    down = int(series["y"] - series["bbox_size"] * bbox_down_ratio)

                    left = max(0, left)
                    down = max(0, down)

                    img = img[down:top, left:right]
                    img_filter = img.copy()
                    for player_id in [1, 2]:
                        if series[f"nfl_player_id_{player_id}"] == "G":
                            continue
                        if np.isnan(series[f"left_{player_id}"]):
                            continue

                        if series[f"nfl_player_id_2"] == "G":
                            box_color = G
                        else:
                            box_color = CONTACT
                        box_left = max(0, int(series[f"left_{player_id}"]) - left)
                        box_right = max(min(img.shape[1], int(series[f"left_{player_id}"] + series[f"width_{player_id}"]) - left), 0)
                        box_top = max(0, int(series[f"top_{player_id}"]) - down)
                        box_down = max(min(img.shape[0], int(series[f"top_{player_id}"] + series[f"height_{player_id}"]) - down), 0)
                        cv2.rectangle(
                            img_filter,
                            (box_left, box_top),
                            (box_right, box_down),
                            box_color,
                            thickness=-1,
                        )
                        # img_filter[
                        #     int(series[f"top_{player_id}"]):int(series[f"top_{player_id}"] + series[f"height_{player_id}"])+1,
                        #     int(series[f"left_{player_id}"]):int(series[f"left_{player_id}"] + series[f"width_{player_id}"])+1,
                        # ] += np.array(box_color, dtype=np.uint8)

                    img = cv2.addWeighted(src1=img, alpha=0.5, src2=img_filter, beta=0.5, gamma=0)
                    # img = (img*0.7 + img_filter*0.3).astype(np.uint8)

                    img = cv2.resize(img, dsize=output_size)
                    out_fname = f"{output_dir}/{game_play}/{view}/{key[0]}_{key[1]}_{frame}.jpg"
                    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
                    cv2.imwrite(out_fname, img)
            vidcap.release()


if __name__ == "__main__":
    main()