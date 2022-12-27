
import pandas as pd
import numpy as np
import cv2
import tqdm

VIDEO_FPS = 25
VIDEO_LENGTH = 60*120
NOT_RANGE_BETWEEN_START_AND_END = -1
EVENT_DICT = {
    "challenge": 0,
    "play": 1,
    "throwin": 2
}
tolerances = {
    "challenge": [0.3, 0.4, 0.5, 0.6, 0.7],
    "play": [0.15, 0.20, 0.25, 0.30, 0.35],
    "throwin": [0.15, 0.20, 0.25, 0.30, 0.35],
}


def get_label_ary(df, fps=VIDEO_FPS, video_sec=VIDEO_LENGTH):
    df["group"] = (df["event"] == "start").cumsum()
    label_ary = np.tile(NOT_RANGE_BETWEEN_START_AND_END, (int(video_sec*fps), 3)).astype(np.float32)
    print(len(label_ary))
    for _, w_df in df.groupby("group"):
        start_time = w_df["time"].values[0]
        end_time = w_df["time"].values[-1]
        start_time_idx = int(start_time * fps)
        end_time_idx = int(end_time * fps)
        label_ary[start_time_idx:end_time_idx] = 0

        w_df = w_df.iloc[1:-1]  # drop start/end
        events = w_df["event"].values
        times = w_df["time"].values

        for i in range(len(w_df)):
            event = events[i]
            event_idx = EVENT_DICT[event]
            event_time = times[i]
            for tolerance in tolerances[event]:
                v_min = event_time - tolerance
                v_max = event_time + tolerance
                v_min_idx = int(v_min * fps)
                v_max_idx = int(v_max * fps)
                label_ary[v_min_idx:v_max_idx, event_idx] += 0.2
    return label_ary


def extract_image(base_dir, output_dir, video_id, labels):
    filename = f"{base_dir}/{video_id}.mp4"
    cap_file = cv2.VideoCapture(filename)
    frame_count = int(cap_file.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_index in tqdm.tqdm(range(frame_count)):
        ret, frame = cap_file.read()
        # frame = cv2.resize(frame, frame_size)
        # frame = frame.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        # if labels[frame_index, 0] >= 0 and labels[frame_index, 1] >= 0 and labels[frame_index, 2] >= 0:
        cv2.imwrite(f"{output_dir}/{video_id}_frame_{frame_index}.jpg", frame)


def main():
    input_dir = '../../data/dfl-bundesliga-data-shootout'
    df = pd.read_csv(f'{input_dir}/train.csv')
    base_dir = "../../data/dfl-bundesliga-data-shootout/train"
    output_dir = "../../data/npy/1920x1080"

    for video_id, w_df in df.groupby("video_id"):
        labels = get_label_ary(w_df)
        np.save(f"{output_dir}/{video_id}_label.npy", labels)
        extract_image(base_dir=base_dir,
                      output_dir=output_dir,
                      video_id=video_id,
                      labels=labels)


if __name__ == "__main__":
    main()




