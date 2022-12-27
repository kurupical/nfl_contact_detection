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
from torchvision.models.video import r3d_18, R3D_18_Weights, r2plus1d_18, R2Plus1D_18_Weights, MViT_V2_S_Weights, mvit_v2_s
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import shutil
import mlflow

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

    if debug:
        epochs: int = 2
    else:
        epochs: int = 1

    lr: float = 1e-4
    n_frames_before: int = 3
    n_frames_after: int = 3
    step: int = 1
    extention: str = ".jpg"
    negative_sample_ratio: float = 0.2
    image_path: str = "images_96x96_v2"
    base_dir: str = "../../notebook/20221221"
    model_name: str = "r3d_18"


class NFL2DModel(nn.Module):

    def __init__(self, model_name):
        self.model_2d = timm.create_model(model_name, num_classes=0)

    def forward(self, x):
        """

        :param x: shape (batch_size, frame_size, c, w, h)
        :return:
        """

        x = x.reshape(-1, )


class NFLDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 base_dir: str,
                 logger: Logger,
                 config: Config,
                 test: bool):
        self.df = df
        self.base_dir = base_dir
        self.config = config
        self.test = test
        self._get_item_information(logger)

    def _get_base_dir(self,
                      game_play: str,
                      view: str,
                      id_1: str,
                      id_2: str):
        return f"{self.base_dir}/{game_play}/{view}/{id_1}_{id_2}"

    def _exist_files(self,
                     game_play: str,
                     view: str,
                     id_1: str,
                     id_2: str,
                     frames: List[int]):
        base_dir = self._get_base_dir(game_play, view, id_1, id_2)
        for frame in frames:
            fname = f"{base_dir}_{frame}{self.config.extention}"
            if not os.path.isfile(fname):
                return False
        return True

    def _get_item_information(self, logger: Logger):
        self.items = []
        logger.info("_get_item_information start")

        df = self.df[self.df["target"]]
        failed_1_count = 0
        failed_0_count = 0
        successed_count = 0

        positive_count = 0
        negative_count = 0

        if not self.test:
            df = pd.concat([
                df[df["contact"] == 1],
                df[df["contact"] == 0].iloc[::int(1 / self.config.negative_sample_ratio)]
            ])

        logger.info(df["contact"].value_counts())
        for i in tqdm.tqdm(range(len(df))):
            series = df.iloc[i]
            frame_idx = series.frame
            min_frame_idx = frame_idx - self.config.n_frames_before * self.config.step
            max_frame_idx = frame_idx + self.config.n_frames_after * self.config.step
            frames = np.arange(min_frame_idx, max_frame_idx + 1, self.config.step)

            if not self._exist_files(
                game_play=series.game_play,
                view=series["view"],
                id_1=series.nfl_player_id_1,
                id_2=series.nfl_player_id_2,
                frames=frames
            ):
                if series.contact == 0:
                    failed_0_count += 1
                elif series.contact == 1:
                    failed_1_count += 1
                continue
            successed_count += 1

            if series.contact == 0:
                negative_count += 1
            else:
                positive_count += 1

            self.items.append({
                "contact_id": series.contact_id,
                "game_play": series.game_play,
                "view": series["view"],
                "id_1": series.nfl_player_id_1,
                "id_2": series.nfl_player_id_2,
                "contact": series.contact,
                "frames": frames
            })
            if self.config.debug and i > 300:
                break

        logger.info(f"finished. extracted={len(self.items)} (total={len(df)}, failed: 0={failed_0_count},1={failed_1_count} successed: {successed_count})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]  # {movie_id}/{start_time}

        contact_id = item["contact_id"]
        frames = item["frames"]
        label = item["contact"]
        base_dir = self._get_base_dir(item["game_play"],
                                      item["view"],
                                      item["id_1"],
                                      item["id_2"])

        if self.config.extention == ".jpg":
            frames = np.stack([
                cv2.imread(f"{base_dir}_{frame}.jpg") for frame in frames
            ], axis=0).transpose(3, 0, 1, 2)  # shape = (C, n_frame, H, W)

        return contact_id, torch.Tensor(frames), torch.Tensor([label])

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

    tk0 = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))

    scaler = torch.cuda.amp.GradScaler()
    for bi, data in tk0:
        batch_size = len(data)

        x = data[1].to(device)
        label = data[2].to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred = model(x)
            loss = criterion(pred, label)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scheduler.step()
        scaler.step(optimizer)
        scaler.update()

        loss_score.update(loss.detach().item(), batch_size)
        mlflow.log_metric("train_loss", loss_score.avg)

        tk0.set_postfix(Loss=loss_score.avg,
                        Epoch=epoch,
                        LR=optimizer.param_groups[0]['lr'])

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
            label = data[2].to(device)

            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = criterion(pred, label)

            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)

            contact_ids.extend(contact_id)
            preds.extend(torch.sigmoid(pred).detach().cpu().numpy())
            labels.extend(label.detach().cpu().numpy())

            del x, label, pred

    preds = np.concatenate(preds, axis=0).astype(np.float16)
    labels = np.concatenate(labels, axis=0).astype(np.float16)

    df_ret = pd.DataFrame({
        "contact_id": contact_ids,
        "score": preds,
        "label": labels
    })
    return df_ret, loss_score.avg


def get_model(model_name):
    # https://pytorch.org/vision/stable/models.html
    if model_name == "r3d_18":
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
        model.fc = nn.Linear(512, 1)
        return model
    if model_name == "r2plus1d_18":
        weights = R2Plus1D_18_Weights.DEFAULT
        model = r2plus1d_18(weights=weights)
        model.fc = nn.Linear(512, 1)
        return model
    if model_name == "mvit_v2_s":
        weights = MViT_V2_S_Weights.DEFAULT
        model = mvit_v2_s(weights=weights)
        model.fc = nn.Linear(512, 1)
        return model
    raise ValueError(model_name)


def main(config):
    output_dir = f"../../output/cnn_3d/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}_{config.exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(__file__, output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = config.base_dir
    df = pd.read_feather(f"{base_dir}/data/gps.feather")
    logger = get_logger(output_dir)
    logger.info("start!")
    gkfold = GroupKFold(5)

    df_label = pd.read_csv("../../input/nfl-player-contact-detection/train_labels.csv")
    if config.debug:
        df_label = df_label.iloc[:150000]

    model = get_model(config.model_name)

    for train_idx, val_idx in gkfold.split(df_label, groups=df_label["game_play"].values):
        df_label_train = df_label.iloc[train_idx]
        df_label_val = df_label.iloc[val_idx]
        df_train = df[df["game_play"].isin(df_label_train["game_play"].values)]
        df_val = df[df["game_play"].isin(df_label_val["game_play"].values)]
        break

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
    df_val_dataset = pd.DataFrame(val_dataset.items)
    df_merge = pd.merge(
        df_label_val[["contact_id", "contact"]],
        df_val_dataset[["contact_id", "contact"]].rename(columns={"contact": "pred"}),
        how="left"
    ).fillna(0).sort_values("contact", ascending=False).drop_duplicates("contact_id")
    possible_score = matthews_corrcoef(df_merge["contact"].values, df_merge["pred"].values)
    logger.info(f"possible MCC score: {possible_score}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=50, num_training_steps=100000)

    results = []
    mlflow.set_tracking_uri('../../mlruns/')

    with mlflow.start_run(run_name=config.exp_name):
        for k, v in config.__dict__.items():
            mlflow.log_param(k, v)
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

                for th in np.arange(0, 1, 0.05):
                    score = matthews_corrcoef(df_score["contact"].values, df_score["score"].values > th)

                    logger.info(f"th={th} func={func}: score={score}")
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
            mlflow.log_param("threshold", best_th)
            mlflow.log_param("func", best_func.__name__)

            pd.DataFrame(results).to_csv(f"{output_dir}/results.csv", index=False)


if __name__ == "__main__":
    # for image_path in ["images_96x96_v3", "images_96x96_v4"]:
    #     for step in [2]:
    #         for negative_sample_ratio in [0.2]:
    #             exp_name = f"step{step}_negative_sample_ratio{negative_sample_ratio}_image_path{image_path}"
    #             config = Config(n_frames_after=15, n_frames_before=15, step=step, exp_name=exp_name, negative_sample_ratio=negative_sample_ratio)
    #             main(config)
    #
    # for image_path in ["images_96x96_v4"]:
    #     for step in [2]:
    #         for negative_sample_ratio in [0.2]:
    #             exp_name = f"step{step}_negative_sample_ratio{negative_sample_ratio}_image_path{image_path}"
    #             config = Config(n_frames_after=32, n_frames_before=32, step=step, exp_name=exp_name, negative_sample_ratio=negative_sample_ratio)
    #             main(config)
    #
    # for image_path in ["images_96x96_v5"]:
    #     for step in [2]:
    #         for negative_sample_ratio in [0.2]:
    #             exp_name = f"step{step}_negative_sample_ratio{negative_sample_ratio}_image_path{image_path}"
    #             config = Config(n_frames_after=15, n_frames_before=15, step=step, exp_name=exp_name, negative_sample_ratio=negative_sample_ratio)
    #             main(config)

    for image_path in ["images_128x128"]:
        exp_name = f"3d_{image_path}"
        config = Config(n_frames_after=15, n_frames_before=15, step=2, exp_name=exp_name, negative_sample_ratio=0.2, image_path=image_path)
        main(config)