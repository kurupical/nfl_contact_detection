import cv2
import pandas as pd
import os
import numpy as np
import tqdm


output_dir = f"../../output/preprocess/images/images_full"
traintest = "train"


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

    df_labels = pd.read_csv(f"{base_dir}/train_labels.csv", parse_dates=["datetime"])

    game_plays = df_labels["game_play"].drop_duplicates().values

    for game_play in tqdm.tqdm(game_plays):
        for view in ["Endzone", "Sideline"]:
            video_path = f"{base_dir}/{traintest}/{game_play}_{view}.mp4"
            vidcap = cv2.VideoCapture(video_path)
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame in range(frame_count):
                it_worked, img = vidcap.read()
                out_fname = f"{output_dir}/{game_play}/{view}/{frame}.jpg"
                os.makedirs(os.path.dirname(out_fname), exist_ok=True)
                cv2.imwrite(out_fname, img)
            vidcap.release()

if __name__ == "__main__":
    main()