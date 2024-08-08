from net import st_gcn
from Client import Client
from Feeder import Feeder
import torch
import copy
from Net_utils import *
# from feeder_kinetics import Feeder_kinetics

test_feeder_args = {
    "data_path": "data/NTU-RGB-D/xview/val_data.npy",
    "label_path": "data/NTU-RGB-D/xview/val_label.pkl",
}

test_args = {
    "phase": "test",
    "device": 0,
    "test_batch_size": 64,
}


class Server:
    def __init__(self, client_num=10) -> None:
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = st_gcn.Model(
        in_channels=3,
        num_class=60,  # Ensure this matches across Server and Client
        dropout=0.5,
        edge_importance_weighting=True,
        graph_args={"layout": "openpose", "strategy": "spatial"},
        ).to(self.dev)

        self.test_loader = torch.utils.data.DataLoader(
            dataset= Feeder(**test_feeder_args),
            batch_size=test_args["test_batch_size"],
            shuffle=False,
            num_workers=2,
        )

        self.clients = [Client() for _ in range(client_num)]
        for client in self.clients:
            client.set_data()

    def train_one_epoch(self):
        for client in self.clients:
            client.train_one_epoch()
        self.aggregate()
        acc_top1, acc_top5 = self.validation()
        return acc_top1, acc_top5

    def aggregate(self):
        averaged_paras = copy.deepcopy(self.model.state_dict())
        # 置0
        total_train_data_num = 0
        for var in averaged_paras:
            averaged_paras[var] = 0

        # 求数据总量
        for client in self.clients:
            total_train_data_num += len(client.train_loader)

        # 联邦平均
        for client in self.clients:  # client_models
            # 按照client的数据量分配权重
            client_data_num = len(client.train_loader)
            client_param_weights = client_data_num / total_train_data_num
            # 加权求和
            for var in averaged_paras:
                averaged_paras[var] += (
                    client.model.state_dict()[var] * client_param_weights
                )
        # 更新全局模型
        self.model.load_state_dict(averaged_paras)

        # 更新局部模型
        for client in self.clients:
            client.model.load_state_dict(averaged_paras)
            
    def validation(self):
        self.model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        with torch.no_grad():
            for data, label in self.test_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                outputs = self.model(data)
                _, predicted_top1 = torch.max(outputs.data, 1)
                _, predicted_top5 = outputs.data.topk(5, dim=1, largest=True, sorted=True)

                total += label.size(0)
                correct_top1 += (predicted_top1 == label).sum().item()
                
                # For top-5 accuracy
                for i in range(label.size(0)):
                    if label[i] in predicted_top5[i]:
                        correct_top5 += 1

        accuracy_top1 = 100 * correct_top1 / total
        accuracy_top5 = 100 * correct_top5 / total
        return accuracy_top1, accuracy_top5

    def train_with_save(self, epoch_num=15):
        with open("server_kn_log.txt", "w") as f:
            f.close()

        for i in range(epoch_num):
            acc_top1, acc_top5 = self.train_one_epoch()
            train_one_epoch_log = "epoch: {}, acc_top1: {:.2f}%, acc_top5: {:.2f}%".format(i, acc_top1, acc_top5)
            print(train_one_epoch_log)
            with open("server_log_kn.txt", "a") as f:
                f.write(train_one_epoch_log + "\n")
            # 每1个epoch保存模型
            if i % 1 == 0:
                torch.save(self.model.state_dict(), "st_gcn_glob_kn.pth")

if __name__ == "__main__":
    a = Server()
    a.train_with_save()
