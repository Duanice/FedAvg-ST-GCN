import numpy as np
import pickle
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from net import st_gcn
from Feeder import Feeder
from Net_utils import *

# optim
optim_arg = {
    "optimizer": "SGD",
    "weight_decay": 0.0001,
    "base_lr": 0.1,
    "step": [10, 50],
}

training_args = {
    "device": [0],
    "batch_size": 32,  # 调整batch size
    "test_batch_size": 32,  # 调整batch size
    "num_epoch": 80,
}

train_feeder_args = {
    "data_path": "data/NTU-RGB-D/xsub/train_data.npy",
    "label_path": "data/NTU-RGB-D/xsub/train_label.pkl",
}


test_feeder_args = {
    "data_path": "data/NTU-RGB-D/xsub/val_data.npy",
    "label_path": "data/NTU-RGB-D/xsub/val_label.pkl",
}

test_args = {
    "phase": "test",
    "device": 0,
    "test_batch_size": 32,
}

class Client:
    def __init__(self) -> None:
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = torch.nn.CrossEntropyLoss().to(self.dev)
        self.meta_info = {"epoch": 0, "iter": 0}
        self.iter_info = {"loss": 0.0, "lr": 0.0}
        self.epoch_info = {"mean_loss": 0.0}

        self.model = st_gcn.Model(
            in_channels=3,
            num_class=60,
            dropout=0.5,
            edge_importance_weighting=True,
            graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        ).to(self.dev)

        self.test_loader = DataLoader(
            dataset=Feeder(**test_feeder_args),
            batch_size=test_args["test_batch_size"],
            shuffle=False,
            num_workers=2,
        )

    def set_data(self, num_workers=2):
        self.train_loader = DataLoader(
            dataset=Feeder(**train_feeder_args),
            batch_size=training_args["batch_size"],  # 64
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

    def load_optimizer(self, optimizer="SGD"):
        optim_arg["optimizer"] = optimizer
        if optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optim_arg["base_lr"],
                momentum=0.9,
                weight_decay=optim_arg["weight_decay"],
            )
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optim_arg["base_lr"],
                weight_decay=optim_arg["weight_decay"],
            )
        else:
            raise ValueError()

    def adjust_lr(self):
        if optim_arg["optimizer"] == "SGD" and optim_arg["step"]:
            lr = optim_arg["base_lr"] * (
                0.1 ** np.sum(self.meta_info["epoch"] >= np.array(optim_arg["step"]))
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.lr = lr
        else:
            self.lr = optim_arg["base_lr"]

    def get_iter_info(self):  # batch
        info = ""
        for k, v in self.iter_info.items():
            if isinstance(v, float):
                info += " | {}: {:.4f}".format(k, v)
            else:
                info += " | {}: {}".format(k, v)
        return info

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            print("\t{}: {}".format(k, v))

    def get_weights(self):
        return self.model.state_dict()

    def validation(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in self.test_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def train_one_epoch(self):
        self.model.train()
        self.load_optimizer()
        self.adjust_lr()
        loader = self.train_loader
        loss_value = []
        tbar = tqdm(loader)

        for data, label in tbar:
            data = data.float().to(self.dev)  # 使用GPU
            label = label.long().to(self.dev)  # 使用GPU

            output = self.model(data)
            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info["loss"] = loss.data.item()
            self.iter_info["lr"] = "{:.6f}".format(self.lr)
            loss_value.append(self.iter_info["loss"])

            tbar.set_description("loss: {:.4f}".format(self.iter_info["loss"]))
            self.meta_info["iter"] += 1

        self.epoch_info["mean_loss"] = np.mean(loss_value)
        self.show_epoch_info()
        acc = self.validation()
        return self.epoch_info["mean_loss"], acc


if __name__ == "__main__":
    epoch = 10
    c = Client()
    c.set_data()

    with open("client_log.txt", "w") as f:
        f.close()

    for _ in range(epoch):
        mean_loss, acc = c.train_one_epoch()
        print("epoch ", epoch, "mean_loss: ", mean_loss, "acc: ", acc)
        with open("client_log.txt", "a") as f:
            f.write(
                "epoch "
                + str(epoch)
                + " mean_loss: "
                + str(mean_loss)
                + " acc: "
                + str(acc)
                + "\n"
            )
