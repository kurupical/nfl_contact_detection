import pandas as pd
import numpy as np
import tqdm
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import copy
import os
import shutil
from datetime import datetime as dt
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from logging import Logger, StreamHandler, Formatter, FileHandler
import logging
pd.set_option("max_row", 1000)
pd.set_option("max_column", 200)

debug = False

def sin(x):
    ans = np.sin(x / 180 * np.pi)
    return ans

def cos(x):
    ans = np.cos(x / 180 * np.pi)
    return ans


def get_logger(output_dir=None, logging_level=logging.INFO):
    formatter = Formatter("%(asctime)s|%(levelname)s| %(message)s")
    logger = Logger(name="log")
    handler = StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging_level)
    logger.addHandler(handler)
    if output_dir is not None:
        now = dt.now().strftime("%Y%m%d%H%M%S")
        file_handler = FileHandler(f"{output_dir}/{now}.txt")
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class LGBMModel:
    def __init__(self,
                 output_dir: str,
                 logger: Logger,
                 exp_id: str,
                 debug: bool = False,
                 params: dict = None):
        if params is None:
            self.logger = logger
            self.params = {
                'objective': 'binary',
                'metrics': 'auc',
                'num_leaves': 32,
                'max_depth': -1,
                'bagging_fraction': 0.7,  # 0.5,
                'feature_fraction': 0.7,
                'bagging_seed': 0,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'random_state': 0,
                'verbosity': -1,
                "n_estimators": 20000,
                "early_stopping_rounds": 100,
                "learning_rate": 0.1,
            }
        self.debug = debug
        self.output_dir = output_dir
        self.exp_id = exp_id
        self.feature_dir = "../../output/preprocess/feature"
        self.model_dir = f"{output_dir}/model.txt"
        self.drop_columns = [
            "contact_id", "game_play", "datetime", "team_1", "position_1", "team_2", "position_2",
            "nfl_player_id_1", "nfl_player_id_2",
            "game_key"
        ]

    def feature_engineering(self,
                            df: pd.DataFrame,
                            inference: bool = True):
        # all
        # df = df[df["distance"].fillna(0) <= 2]
        feature_path = f"{self.feature_dir}/{os.path.basename(__file__).replace('.py', '')}/feature_len{len(df)}.feather"
        if os.path.isfile(feature_path) and not inference and not self.debug:
            self.logger.info("load from feature_dir")
            return pd.read_feather(feature_path)

        self.logger.info("FE1: all features")

        self.logger.info(f"[aggregate view]before: {len(df)}")
        helmet_columns = [
            "left_1", "width_1", "top_1", "height_1", "x_1", "y_1",
            "left_2", "width_2", "top_2", "height_2", "x_2", "y_2"
        ]
        df_endzone = df[df["view"] == "Endzone"].drop("view", axis=1)
        df_sideline = df[df["view"] == "Sideline"].drop("view", axis=1)
        df_endzone.columns = [f"Endzone_{col}" if col in helmet_columns else col for col in df_endzone.columns]
        df_sideline.columns = [f"Sideline_{col}" if col in helmet_columns else col for col in df_sideline.columns]

        df = df[["contact_id"]].drop_duplicates()
        df = pd.merge(df, df_endzone, how="left")
        df = pd.merge(df, df_sideline, how="left")
        self.logger.info(f"[aggregate view]after: {len(df)}")

        for col in ["orientation", "direction"]:
            for player_id in [1, 2]:
                col_name = f"{col}_{player_id}"
                df[f'{col_name}_sin'] = df[col_name].apply(lambda x: sin(x))
                df[f'{col_name}_cos'] = df[col_name].apply(lambda x: cos(x))

            for col2 in ["acceleration", "sa"]:
                for col3 in ["sin", "cos"]:
                    for player_id in [1, 2]:
                        col_name = f"{col}_{col2}_{col3}_{player_id}"
                        df[col_name] = df[f"{col2}_{player_id}"] * df[f"{col}_{player_id}_{col3}"]

        df["distance"] = np.sqrt(
            (df["x_position_1"].values - df["x_position_2"]) ** 2 + \
            (df["y_position_1"].values - df["y_position_2"]) ** 2
        )

        for view in ["Endzone", "Sideline"]:
            df[f"{view}_distance_helmet"] = np.sqrt(
                (df[f"{view}_x_1"].values - df[f"{view}_x_2"]) ** 2 + \
                (df[f"{view}_y_1"].values - df[f"{view}_y_2"]) ** 2
            )
        df[f"distance_helmet_mean"] = df[["Endzone_distance_helmet", "Sideline_distance_helmet"]].mean(axis=1)

        df["move_sensor"] = df["distance_1"] + df["distance_2"]

        df["is_same_team"] = df["team_1"] == df["team_2"]
        df["is_G"] = df["nfl_player_id_2"] == "G"
        for col in ["orientation", "direction"]:
            for col2 in ["acceleration", "sa"]:
                for col3 in ["sin", "cos"]:
                    col_name = f"{col}_{col2}_{col3}"
                    df[f"{col_name}_diff"] = df[f"{col_name}_1"] - df[f"{col_name}_2"]

        # group by pair
        df_rets = []
        lag_columns = [
            "distance",
            "move_sensor",
            "distance_helmet_mean",
            "orientation_acceleration_sin_diff",
            "orientation_acceleration_cos_diff",
            "orientation_sa_sin_diff",
            "orientation_sa_cos_diff",
            "direction_acceleration_sin_diff",
            "direction_acceleration_cos_diff",
            "direction_sa_sin_diff",
            "direction_sa_cos_diff",
        ]
        long_lag_columns = [
            "distance_helmet_mean",
            "move_sensor",
            "distance"
        ]

        for view in ["Endzone", "Sideline"]:
            lag_columns.extend([
                f"{view}_x_1",
                f"{view}_x_2",
                f"{view}_y_1",
                f"{view}_y_2",
            ])
        self.logger.info("groupby features")

        # TODO: speedup
        for _, w_df in tqdm.tqdm(df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])):
            # lag
            for lag in [1, 5, 10, 20, -1, -5, -10, -20]:
                cols = [f"{lag_column}_lag{lag}" for lag_column in lag_columns]
                w_df[cols] = w_df[lag_columns].diff(lag)
            df_rets.append(w_df)

            for lag in [-120, -60, 60, 120]:
                cols = [f"{lag_column}_lag{lag}" for lag_column in long_lag_columns]
                w_df[cols] = w_df[long_lag_columns].diff(lag)
            df_rets.append(w_df)
        df_rets = pd.concat(df_rets).sort_index().reset_index(drop=True)
        for view in ["Endzone", "Sideline"]:
            df_rets[f"{view}_move_helmet_1"] = np.sqrt(df_rets[f"{view}_x_1_lag1"].diff() ** 2 + df_rets[f"{view}_y_1_lag1"].diff() ** 2)
            df_rets[f"{view}_move_helmet_2"] = np.sqrt(df_rets[f"{view}_x_2_lag1"].diff() ** 2 + df_rets[f"{view}_y_2_lag1"].diff() ** 2)
            df_rets[f"{view}_move_helmet"] = df_rets[[f"{view}_move_helmet_1", f"{view}_move_helmet_2"]].mean(axis=1)
        df_rets["move_helmet"] = df_rets[["Endzone_move_helmet", "Sideline_move_helmet"]].mean(axis=1)

        # # 2nd groupby -> memory leakなったら面倒なので後で。。
        # lag_columns2 = ["move_helmet"]
        # df_rets2 = []
        # for _, w_df in tqdm.tqdm(df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])):
        #     # lag
        #     for lag in [1, 5, 10, 20, 60]:
        #         cols = [f"{lag_column}_lag{lag}" for lag_column in lag_columns2]
        #         w_df[cols] = w_df[lag_columns2].diff(lag)
        #     df_rets2.append(w_df)
        # df_rets2 = pd.concat(df_rets2).sort_index().reset_index(drop=True)

        agg_cols = [
            "distance", "move_helmet", "move_sensor", "distance_helmet_mean",
            "distance_lag1", "distance_lag20",
            "move_sensor_lag1", "move_sensor_lag20",
            # "move_helmet_lag1", "move_helmet_lag20", "move_helmet_lag60",
            "distance_helmet_mean_lag1", "distance_helmet_mean_lag20",
        ]
        self.logger.info("aggregate features")
        for agg_col in tqdm.tqdm(agg_cols):
            col_name = f"{agg_col}_mean"
            df_rets[col_name] = df_rets.groupby("game_play")[agg_col].transform("mean")
            df_rets[f"diff_{col_name}"] = df_rets[agg_col] - df_rets[col_name]
            df_rets[f"div_{col_name}"] = df_rets[agg_col] / df_rets[col_name]

        if not inference:
            self.logger.info("save feather")
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            df_rets.to_feather(feature_path)

        return df_rets

    def train(self,
              df: pd.DataFrame,
              df_label: pd.DataFrame = None):

        gkfold = GroupKFold(5)
        df_fe = self.feature_engineering(df, inference=False)
        if df_label is None:
            if self.debug:
                df_label = df
            else:
                df_label = pd.read_csv("../../input/nfl-player-contact-detection/train_labels.csv")

        self.logger.info((df_fe.isnull().sum() / len(df_fe)).sort_values())

        for train_idx, val_idx in gkfold.split(df_label, groups=df_label["game_play"].values):
            df_label_train = df_label.iloc[train_idx]
            df_label_val = df_label.iloc[val_idx]
            df_train = df_fe[df_fe["game_play"].isin(df_label_train["game_play"].values)]
            df_val = df_fe[df_fe["game_play"].isin(df_label_val["game_play"].values)]
            df_test = df[df["game_play"].isin(df_label_val["game_play"].values)]
            break

        df_merge = pd.merge(
            df_label_val[["contact_id", "contact"]],
            df_val[["contact_id", "contact"]].rename(columns={"contact": "pred"}),
            how="left"
        ).fillna(0).sort_values("contact", ascending=False).drop_duplicates("contact_id")
        possible_score_all = matthews_corrcoef(df_merge["contact"].values, df_merge["pred"].values == 1)
        self.logger.info(f"possible MCC score: {possible_score_all}")

        dataset_train = lgb.Dataset(df_train.drop(self.drop_columns + ["contact"], axis=1), label=df_train["contact"])
        dataset_val = lgb.Dataset(df_val.drop(self.drop_columns + ["contact"], axis=1), label=df_val["contact"])

        lgb.register_logger(self.logger)
        self.model = lgb.train(
            copy.copy(self.params),
            dataset_train,
            valid_sets=[dataset_train, dataset_val],
            verbose_eval=100,
        )

        self.model.save_model(self.model_dir)

        # inference
        self.model = lgb.Booster(model_file=self.model_dir)
        pred, contact_id = self.predict(df_test)

        df_pred = pd.DataFrame({
            "contact_id": contact_id,
            "pred": pred
        })

        df_merge = pd.merge(
            df_label_val, df_pred, how="left"
        )
        df_merge["pred"] = df_merge["pred"].fillna(0)

        pred = df_merge["pred"].values
        contact = df_merge["contact"].values

        self.logger.info(f"auc: {roc_auc_score(contact, pred)}")
        self.logger.info("------- MCC -------")
        for th in np.arange(0, 1, 0.05):
            self.logger.info(f"th={th}: {matthews_corrcoef(contact, pred > th)}")

        pd.DataFrame({
            "col": self.model.feature_name(),
            "imp": self.model.feature_importance() / self.model.feature_importance().sum()
        }).sort_values("imp", ascending=False).to_csv(f"{self.output_dir}/feature_importance.csv", index=False)
        pd.DataFrame({
            "contact_id": df_merge["contact_id"].values,
            "score": pred
        }).to_csv(f"{self.output_dir}/pred.csv", index=False)

    def predict(self,
                df: pd.DataFrame):
        df = self.feature_engineering(df)
        return self.model.predict(df[self.model.feature_name()]), df["contact_id"].values

def main():

    output_dir = f"../../output/lgbm/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(__file__, output_dir)
    logger = get_logger(output_dir)

    df = pd.read_feather("../../output/preprocess/master_data_v2/gps.feather")
    if debug:
        df = df.head(300000)

    model = LGBMModel(output_dir=output_dir, logger=logger, exp_id="exp001", debug=debug)
    model.train(df)

if __name__ == "__main__":
    main()
