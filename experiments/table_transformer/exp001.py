import torch
from torch import nn
import timm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime as dt
from logging import Logger, StreamHandler, Formatter, FileHandler
from sklearn.model_selection import GroupKFold
import logging
import dataclasses
import tqdm
import numpy as np
from typing import List
from transformers import get_linear_schedule_with_warmup
import cv2
from torchvision.io.video import read_video
try:
    from torchvision.models.video import r3d_18, R3D_18_Weights
    import mlflow
except Exception as e:
    print(e)
    from torchvision.models.video import r3d_18
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
import shutil
import torch.nn.functional as F
import pickle
from typing import Tuple

torch.backends.cudnn.benchmark = True

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


@dataclasses.dataclass
class Config:
    exp_name: str
    debug: bool = False

    epochs: int = 5
    if debug:
        epochs: int = 1

    lr: float = 1e-4
    weight_decay: float = 0.001
    n_frames: int = 31
    n_predict_frames: int = 1

    if n_frames % 2 == 0 or n_predict_frames % 2 == 0:
        raise ValueError
    step: int = 3
    extention: str = ".npy"
    negative_sample_ratio_close: float = 0.05
    negative_sample_ratio_far: float = 0.05
    base_dir: str = "../../output/preprocess/images"
    data_dir: str = f"../../output/preprocess/master_data_v2"
    image_path: str = "images_128x96"
    img_shape: Tuple[int, int] = (96, 128)
    gradient_clipping: float = 1
    exist_image_threshold: float = 0.1
    data_per_epoch: float = 1
    grayscale: bool = False
    batch_size: int = 256
    use_data_step: int = 1

    num_training_steps: int = 100000

    # transformer
    hidden_size_transformer: int = 178  # TODO: exp002に合わせて
    num_layer_transformer: int = 2
    nhead: int = 2
    num_layer_rnn: int = 1
    feature_dir: str = "../../output/preprocess/feature/exp002/feature_len9443236.feather"
    max_length: int = 192

    # 2d_cnn
    model_name: str = "cnn_3d_r3d_18"
    seq_model: str = "flatten"
    dropout_seq: float = 0.2
    activation: nn.Module = nn.Identity

    kernel_size_conv1d: int = 3
    stride_conv1d: int = 1
    hidden_size_1d: int = 32

    submission_mode: bool = False



class NFLDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 base_dir: str,
                 logger: Logger,
                 config: Config,
                 test: bool,
                 image_dict: dict = None):
        self.base_dir = base_dir
        self.config = config
        self.test = test
        self.exist_files = set()
        self.image_dict = image_dict
        self._get_item_information(df, logger)

    def _get_base_dir(self,
                      game_play: str,
                      view: str,
                      id_1: str,
                      id_2: str):
        return f"{self.base_dir}/{game_play}/{view}/{id_1}_{id_2}"

    def _get_key(self,
                 game_play,
                 view,
                 id_1,
                 id_2,
                 frame):
        if self.image_dict is not None:
            # for submission
            return f"{game_play}_{view}_{id_1}_{id_2}_{frame}"
        else:
            # for local training
            base_dir = self._get_base_dir(game_play, view, id_1, id_2)
            return f"{base_dir}_{frame}{self.config.extention}"

    def _exist_file(self,
                    game_play,
                    view,
                    id_1,
                    id_2,
                    frame):
        if self.image_dict is not None:
            # for submission
            return self._get_key(game_play, view, id_1, id_2, frame) in self.image_dict
        else:
            # for local training
            return os.path.isfile(self._get_key(game_play, view, id_1, id_2, frame))

    def _exist_files(self,
                     game_play: str,
                     id_1: str,
                     id_2: str,
                     frames: List[int]):
        count = 0

        for view in ["Sideline", "Endzone"]:
            base_dir = self._get_base_dir(game_play, view, id_1, id_2)
            for frame in frames:
                if self._exist_file(game_play, view, id_1, id_2, frame):
                    count += 1
                if count > self.config.n_frames * 2 * self.config.exist_image_threshold:
                    return True
        return False

    def _exist_center_files(self,
                            game_play: str,
                            id_1: str,
                            id_2: str,
                            frames: List[int]):
        # 予測対象のframeはSideline / Endline どっちかにはファイルがいてほしい
        for frame in frames:
            ret = False
            for view in ["Sideline", "Endzone"]:
                if self._exist_file(game_play, view, id_1, id_2, frame):
                    ret = True
            if not ret:
                return False
        return True

    def _get_item_information(self, df: pd.DataFrame, logger: Logger):
        self.items = []
        logger.info("_get_item_information start")

        failed_count = 0
        np.random.seed(0)

        drop_columns = [
            "contact_id", "game_play", "datetime", "team_1", "position_1", "team_2", "position_2",
            "nfl_player_id_1", "nfl_player_id_2",
            "game_key", "contact"
        ]
        label_sum = 0

        for key, w_df in tqdm.tqdm(df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])):
            contacts_ = w_df["contact"].values

            contact_ids = [""] * config.max_length
            contact_ids[:len(w_df)] = w_df["contact_id"].values.tolist()
            w_df = w_df.drop(drop_columns, axis=1).fillna(0).replace(np.inf, 0).replace(-np.inf, 0)

            feature_ary = np.zeros((config.max_length, config.hidden_size_transformer))
            feature_ary[:len(w_df), :] = w_df.values

            mask = np.ones(config.max_length)
            mask[:len(contacts_)] = 0

            contacts = np.zeros(config.max_length)
            contacts[:len(contacts_)] = contacts_
            label_sum += contacts.sum()

            self.items.append({
                "contact_id": contact_ids,
                "contact": contacts,
                "feature": feature_ary,
                "mask": mask
            })

        logger.info(f"finished. extracted={len(self.items)} (total_df={len(df)}, label_sum={label_sum})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        contact_id = item["contact_id"]
        feature = item["feature"]
        mask = item["mask"]
        labels = item["contact"]

        return contact_id, torch.Tensor(feature), torch.Tensor(mask), torch.Tensor(labels)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch, config):
    model.train()
    loss_score = AverageMeter()

    data_length = int(len(dataloader) * config.data_per_epoch)
    tk0 = tqdm.tqdm(enumerate(dataloader), total=data_length)

    scaler = torch.cuda.amp.GradScaler()
    count = 0
    loss_100 = []
    for bi, data in tk0:
        count += 1
        batch_size = len(data)

        x = data[1].to(device)
        mask = data[2].to(device)
        label = data[3].to(device)
        optimizer.zero_grad()

        mask_flat = mask.flatten()

        label = label.flatten()
        label = label[mask_flat == 0]

        with torch.cuda.amp.autocast():
            pred = model(x, mask).flatten()
            pred = pred[mask_flat == 0]
            loss = criterion(pred, label)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        scheduler.step()
        scaler.step(optimizer)
        scaler.update()

        loss = loss.detach().item()
        loss_100.append(loss)
        loss_100 = loss_100[-100:]
        loss_score.update(np.mean(loss_100), batch_size)
        mlflow.log_metric("train_loss", loss_score.avg)
        mlflow.log_metric("train_loss_snap", np.mean(loss_100))

        tk0.set_postfix(Loss=loss_score.avg,
                        LossSnap=np.mean(loss_100),
                        Epoch=epoch,
                        LR=optimizer.param_groups[0]['lr'])

        if count > data_length:
            break

    return loss_score.avg


def eval_fn(data_loader, model, criterion, device):
    loss_score = AverageMeter()

    model.eval()
    tk0 = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    preds = []
    contact_ids = []
    labels = []

    with torch.no_grad():
        for bi, data in tk0:
            batch_size = len(data)

            contact_id = data[0]
            x = data[1].to(device)
            mask = data[2].to(device)
            label = data[3].to(device)

            mask_flat = mask.flatten().detach().cpu().numpy()

            contact_id = np.array(contact_id).flatten()
            contact_id = contact_id[mask_flat == 0]

            mask_flat = mask.flatten()
            label = label.flatten()
            label = label[mask_flat == 0]

            with torch.cuda.amp.autocast():
                pred = model(x, mask).flatten()
                pred = pred[mask_flat == 0]
                loss = criterion(pred, label)
            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)

            contact_ids.extend(np.array(contact_id).flatten())
            preds.extend(torch.sigmoid(pred.flatten()).detach().cpu().numpy())
            labels.extend(label.flatten().detach().cpu().numpy())

            del x, label, pred

    preds = np.array(preds).astype(np.float16)
    labels = np.array(labels).astype(np.float16)

    df_ret = pd.DataFrame({
        "contact_id": contact_ids,
        "score": preds,
        "label": labels,
    })
    return df_ret, loss_score.avg


class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.ln1 = nn.LayerNorm(state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.ln2 = nn.LayerNorm(state_size)

    def forward(self, x):
        x = self.lr1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.ln2(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self,
                 config: Config):
        super().__init__()
        self.config = config
        self.bn = nn.BatchNorm1d(self.config.hidden_size_transformer)
        transformer_encoder = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size_transformer,
            nhead=self.config.nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            num_layers=self.config.num_layer_transformer,
            encoder_layer=transformer_encoder,
            # norm=nn.LayerNorm(self.config.num_layer_transformer),
        )

        self.rnn = nn.LSTM(
            input_size=self.config.hidden_size_transformer,
            hidden_size=self.config.hidden_size_transformer,
            num_layers=self.config.num_layer_rnn,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(self.config.hidden_size_transformer)
        self.ffn = FFN(self.config.hidden_size_transformer)
        self.fc = nn.LazyLinear(1)

    def forward(self, x, mask):
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.transformer(x)  # (bs, seq_len, n_feature) -> (bs, seq_len, n_feature)
        x = self.rnn(x)[0]  # (bs, seq_len, n_feature) -> (bs, seq_len, n_feature)
        x = self.layer_norm(x)

        # x = self.ffn(x)  # (bs, seq_len, n_feature) -> (bs, seq_len, n_feature)
        x = self.fc(x)  # (bs, seq_len, n_feature) -> (bs, seq_len, 1)
        x = x.squeeze(2)  # (bs, seq_len, 1) -> (bs, seq_len)
        return x


def get_df_from_item(item):
    df = pd.DataFrame({
        "contact_id": item["contact_id"],
        "contact": item["contact"] == 1,
    })
    df["contact"] = df["contact"].astype(int)
    return df

def main(config):
    output_dir = f"../../output/table_transformer/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}_{config.exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(__file__, output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(f"{output_dir}/cfg.pickle", "wb") as f:
        pickle.dump(config, f)

    base_dir = config.base_dir
    df = pd.read_feather(f"{config.data_dir}/gps.feather")

    df_feature = pd.read_feather(config.feature_dir)
    logger = get_logger(output_dir)
    logger.info("start!")
    gkfold = GroupKFold(5)

    df_label = pd.read_csv("../../input/nfl-player-contact-detection/train_labels.csv")
    if config.debug:
        df_label = df_label.iloc[:150000]

    model = TransformerModel(config=config)

    for train_idx, val_idx in gkfold.split(df_label, groups=df_label["game_play"].values):
        df_label_train = df_label.iloc[train_idx]
        df_label_val = df_label.iloc[val_idx]
        df_train = df_feature[df_feature["game_play"].isin(df_label_train["game_play"].values)]
        df_val = df_feature[df_feature["game_play"].isin(df_label_val["game_play"].values)]
        break
    df_merge = pd.merge(
        df_label[["contact_id", "contact"]],
        df[["contact_id", "contact"]].rename(columns={"contact": "pred"}),
        how="left"
    ).fillna(0).sort_values("contact", ascending=False).drop_duplicates("contact_id")
    possible_score_all = matthews_corrcoef(df_merge["contact"].values, df_merge["pred"].values == 1)
    logger.info(f"possible MCC score: {possible_score_all}")

    train_dataset = NFLDataset(
        df=df_train,
        base_dir=f"{base_dir}/{config.image_path}",
        logger=logger,
        config=config,
        test=False
    )

    val_dataset = NFLDataset(
        df=df_val,
        base_dir=f"{base_dir}/{config.image_path}",
        logger=logger,
        config=config,
        test=True
    )

    if config.debug:
        train_dataset.items = train_dataset.items[:200]
        val_dataset.items = val_dataset.items[:200]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=1
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=1
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=50, num_training_steps=config.num_training_steps
    )

    results = []
    mlflow.set_tracking_uri('../../mlruns/')

    try:
        with mlflow.start_run(run_name=config.exp_name):
            for k, v in config.__dict__.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("MCC_all", possible_score_all)
            mlflow.log_metric("MCC_extracted", possible_score_all)
            mlflow.log_param("output_dir", output_dir)
            for epoch in range(config.epochs):
                logger.info(f"===============================")
                logger.info(f"epoch {epoch + 1}")
                logger.info(f"===============================")
                train_loss = train_fn(
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    device,
                    scheduler,
                    epoch,
                    config,
                )

                df_pred, valid_loss = eval_fn(
                    val_loader,
                    model,
                    criterion,
                    device
                )
                df_merge = pd.merge(df_label_val, df_pred, how="left")
                df_merge["score"] = df_merge["score"].fillna(0)

                best_th = -1
                best_score = -1
                best_func = None
                logger.info(f"loss: train {train_loss}, val {valid_loss}")
                logger.info(f"------ MCC ------")
                for func in [np.mean, np.max, np.min]:
                    df_score = df_merge.groupby(["contact_id", "contact"], as_index=False)["score"].apply(func)

                    auc = roc_auc_score(df_score["contact"].values, df_score["score"].values)
                    logger.info(f"\nfunc={func} auc: {auc}")

                    label = df_score["contact"].values
                    pred = df_score["score"].values
                    for th in np.arange(0, 1, 0.05):
                        score = matthews_corrcoef(label, pred > th)

                        logger.info(f"th={th} func={func}: score={score}")
                        logger.info(f"counfusion_matrix: \n{confusion_matrix(label, pred > th)}")
                        if best_score < score:
                            best_th = th
                            best_score = score
                            best_func = func

                logger.info(f"***************** epoch {epoch} *****************")
                logger.info(f"best: {best_score} (th={best_th}, func={best_func})")
                logger.info(f"******************************************")
                df_merge.to_csv(f"{output_dir}/pred_{epoch}.csv", index=False)
                torch.save(model.state_dict(), f"{output_dir}/epoch{epoch}.pth")

                results.append({
                    "epoch": epoch,
                    "score": best_score,
                    "train_loss": train_loss,
                    "val_loss": valid_loss,
                    "th": best_th,
                    "func": best_func
                })
                mlflow.log_metric("val_loss", valid_loss)
                mlflow.log_metric("score", best_score)

                pd.DataFrame(results).to_csv(f"{output_dir}/results.csv", index=False)
            mlflow.log_param("threshold", best_th)
            mlflow.log_param("func", best_func.__name__)
    except Exception as e:
        print(e)
        raise e


if __name__ == "__main__":

    for lr in [1e-3, 3e-3, 1e-4, 3e-4]:
        exp_name = f"transformer_lr{lr}"
        config = Config(exp_name=exp_name, lr=lr)
        main(config)