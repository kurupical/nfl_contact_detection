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
        self.output_dir = output_dir
        self.exp_id = exp_id
        self.feature_dir = "../../output/preprocess/feature"
        self.model_dir = f"{output_dir}/model.txt"
        self.drop_columns = [
            "contact_id", "game_play", "datetime", "team_1", "position_1", "team_2", "position_2",
            "nfl_player_id_1", "nfl_player_id_2",
            "view", "video", "nfl_player_id", "player_label", "team", "diff",
            "jersey_number_1", "jersey_number_2", "game_key", "play_id",
            "play_id_1", "play_id_2",
        ]

    def feature_engineering(self,
                            df: pd.DataFrame,
                            inference: bool = True):
        # all
        feature_path = f"{self.feature_dir}/feature_len{len(df)}.feather"
        if os.path.isfile(feature_path) and not inference:
            self.logger.info("load from feature_dir")
            return pd.read_feather(feature_path)

        self.logger.info("FE1: all features")
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
            "orientation_acceleration_sin_diff",
            "orientation_acceleration_cos_diff",
            "orientation_sa_sin_diff",
            "orientation_sa_cos_diff",
            "direction_acceleration_sin_diff",
            "direction_acceleration_cos_diff",
            "direction_sa_sin_diff",
            "direction_sa_cos_diff",
        ]
        self.logger.info("groupby features")

        # TODO: speedup
        for _, w_df in tqdm.tqdm(df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])):
            # lag
            for col in lag_columns:
                for lag in [1, 5, 10, 20]:
                    w_df[f"{col}_lag{lag}"] = w_df[col].diff(lag)
            df_rets.append(w_df)
        df_rets = pd.concat(df_rets).sort_index().reset_index(drop=True)

        if not inference:
            self.logger.info("save feather")
            df_rets.to_feather(feature_path)

        return df_rets

    def train(self,
              df: pd.DataFrame):

        gkfold = GroupKFold(5)
        df = self.feature_engineering(df, inference=False)

        self.logger.info((df.isnull().sum() / len(df)).sort_values())

        for train_idx, val_idx in gkfold.split(df, groups=df["game_play"].values):
            df_train = df.iloc[train_idx]
            df_val = df.iloc[val_idx]
            break

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
        pred = self.model.predict(df_val[self.model.feature_name()])
        contact = df_val["contact"].values

        self.logger.info(f"auc: {roc_auc_score(contact, pred)}")
        self.logger.info("------- MCC -------")
        for th in np.arange(0, 1, 0.05):
            self.logger.info(f"th={th}: {matthews_corrcoef(contact, pred > th)}")

        pd.DataFrame({
            "col": self.model.feature_name(),
            "imp": self.model.feature_importance() / self.model.feature_importance().sum()
        }).sort_values("imp", ascending=False).to_csv(f"{self.output_dir}/feature_importance.csv", index=False)
        pd.DataFrame({
            "contact_id": df_val["contact_id"].values,
            "score": pred
        }).to_csv(f"{self.output_dir}/pred.csv", index=False)

        # inference test
        self.model = lgb.Booster(model_file=self.model_dir)
        df_test = df.iloc[:1000]
        self.predict(df_test)

    def predict(self,
                df: pd.DataFrame):
        df = self.feature_engineering(df)
        return self.model.predict(df[self.model.feature_name()])

def main():

    output_dir = f"../../output/lgbm/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(__file__, output_dir)
    logger = get_logger(output_dir)

    df = pd.read_feather("../../output/preprocess/master_data/gps.feather")
    # df = pd.read_feather("../../output/preprocess/master_data_v2/gps.feather")
    # df = df.head(300000)

    model = LGBMModel(output_dir=output_dir, logger=logger, exp_id="exp001")
    model.train(df)

if __name__ == "__main__":
    main()
