# NOTE 最终外部使用这几个函数 构成联邦学习的网络流
# 服务器下发模型 send_model_with_number
# 客户端接收模型 recv_model_with_number
# 客户端发送模型 send_model_with_number
# 服务器聚合模型 recv_many_model_with_number

# 注意这里的写法要求每次是接收方先建立监听
# 这里的实现是 联邦服务器发送一次模型 客户端进行一轮训练 再回传模型 服务器再进行聚合
# 服务器会等待所有客户端都发送模型之后再进行聚合 没有做并发处理 没有做超时忽略处理
# 另外服务器和客户端的地址是写死的 没有做IP过滤 理论上任何人都可以连接


from net import st_gcn
import torch
from torch import nn
from Feeder import Feeder
from torch import optim
import numpy as np

import socket
import pickle
from copy import deepcopy
from typing import List, Dict, Any


def initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0, 0.01)
            m.bias.data.zero_()


def set_variables():
    global dev, model, model2  # , fl_server_addr, fl_client_addr, model_dict_size
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = st_gcn.Model(
        in_channels=3,
        num_class=60,
        dropout=0.5,
        edge_importance_weighting=True,
        graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
    ).to(dev)

    model2 = st_gcn.Model(
        in_channels=3,
        num_class=60,
        dropout=0.5,
        edge_importance_weighting=True,
        graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
    ).to(dev)


fl_server_addr = ("", 38291)
fl_client_addr = ("", 38290)


def get_model_size(model: nn.Module):
    ser = pickle.dumps(model.state_dict())
    return len(ser)


def device_size_diff():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_model = st_gcn.Model(
        in_channels=3,
        num_class=60,
        dropout=0.5,
        edge_importance_weighting=True,
        graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
    )
    model_cpu_size = get_model_size(cpu_model)
    gpu_model = cpu_model.to(dev)
    model_gpu_size = get_model_size(gpu_model)
    assert model_cpu_size < model_gpu_size


def model_the_same(net1: nn.Module, net2: nn.Module):
    # 如何在 PyTorch 中比较两个不同模型的参数？ - X-Omics的回答 - 知乎
    # https://www.zhihu.com/question/594626401/answer/2976444325

    # 获取参数
    params1 = list(net1.named_parameters())
    params2 = list(net2.named_parameters())
    # 比较参数
    for p1, p2 in zip(params1, params2):
        if not torch.allclose(p1[1].data, p2[1].data):
            print(p1[0], p2[0], "Parameters are not equal.")
            return False
    print("Parameters are equal.")
    return True
    # for p1, p2 in zip(params1, params2):
    #     if not torch.allclose(p1.data, p2.data):
    #         print("Parameters are not equal.")
    #         return False
    # print("Parameters are equal.")
    # return True


def assign_model():
    # faster
    param_data = pickle.dumps(model.state_dict())
    model_dict = pickle.loads(param_data)
    model2.load_state_dict(model_dict)


def test_assign_model():
    assert get_model_size(model) == get_model_size(model2)
    assign_model()
    assert model_the_same(model, model2)


def assign_model_2():
    torch.save(model.state_dict(), "tmp.pt")
    tmp_state_dict = torch.load("tmp.pt")
    model2.load_state_dict(tmp_state_dict)


def test_speed():
    import time

    start = time.time()
    for _ in range(100):
        assign_model()
    end = time.time()
    as_time_1 = end - start
    print("assign_model 1", as_time_1)

    start = time.time()
    for _ in range(100):
        assign_model_2()
    end = time.time()
    as_time_2 = end - start

    print("assign_model_2", as_time_2)
    print(
        "assign_model_2 is ",
        "faster" if as_time_2 < as_time_1 else "slower",
        " than assign_model",
    )


"""
以上是辅助和检测手段 
以下是网络通信的函数
"""

# 单次通信 服务器等待客户端发送模型


def send_model_to(destination_addr, local_model: nn.Module):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(destination_addr)
        print("sending model")
        data = pickle.dumps(local_model.state_dict())
        s.sendall(data)
    print("model sent")


def recv_model(self_addr, local_model: nn.Module):
    model_dict_size = get_model_size(local_model)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(self_addr)
        s.listen()
        print("waiting for connection")
        conn, addr = s.accept()
        with conn:
            data = conn.recv(model_dict_size)
            model_dict = pickle.loads(data)
            local_model.load_state_dict(model_dict)
        print("model received")


def send_model_with_number(destination_addr, local_model: nn.Module, number: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(destination_addr)
        # NOTE 编号只有两位 目前应该够用
        print("sending model and number")
        data = pickle.dumps({"number": number, "model_dict": local_model.state_dict()})
        s.sendall(data)
    print(f"model {number} sent")


def recv_model_with_number(self_addr, local_model: nn.Module) -> int:
    model_dict_size = get_model_size(local_model)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(self_addr)
        s.listen()
        print("waiting for connection")
        conn, addr = s.accept()
        with conn:
            # 接收编号和模型
            data = conn.recv(32 + model_dict_size)  # NOTE 大小暂时够用
            num_model_struct = pickle.loads(data)
            local_model.load_state_dict(num_model_struct["model_dict"])
        print(f"model from client {num_model_struct['number']} received")
        return num_model_struct["number"]


def recv_many_model(self_addr, local_models: List[nn.Module]):
    # 目前不需要区分客户端编号
    model_dict_size = get_model_size(local_models[0])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(self_addr)
        s.listen()
        print("waiting for connection")
        # 是否需要支持并发
        for i in range(len(local_models)):
            conn, addr = s.accept()
            with conn:
                data = conn.recv(model_dict_size)
                model_dict = pickle.loads(data)
                local_models[i].load_state_dict(model_dict)
        print("models received")


def recv_many_model_with_number(self_addr, local_models: List[st_gcn.Model]):
    # 目前不需要区分客户端编号
    model_dict_size = get_model_size(local_models[0])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(self_addr)
        s.listen()
        print("waiting for connection")
        # 是否需要支持并发
        for _ in range(len(local_models)):
            conn, addr = s.accept()
            with conn:
                # data = conn.recv(32 + model_dict_size)
                data = b""
                while True:
                    packet = conn.recv(32 + model_dict_size)
                    if not packet:
                        break
                    data += packet
                model_num_pack = pickle.loads(data)
                local_models[model_num_pack["number"]].load_state_dict(
                    model_num_pack["model_dict"]
                )
                print(f"model from client {model_num_pack['number']} received")
        print("models all received")


def test_net():
    import time
    import threading

    initialize_weights(model)
    assert not model_the_same(model, model2)

    th_obj = threading.Thread(
        target=recv_model,
        args=(
            fl_server_addr,
            model2,
        ),
    )
    th_obj.start()
    time.sleep(2)
    send_model_to(("127.0.0.1", fl_server_addr[1]), model)
    th_obj.join()

    assert model_the_same(model, model2)


# TODO 加上编号 用于区分客户端 按序号排列 然后测试是否参数相同
# 发送之前还需要确保对应位置的模型是否相同 一边初始化 一边不初始化
def test_many_steam(model_num=10):
    import time
    import threading

    # 先放多个模型用来占位 注意同样的GPU或者同样的CPU
    clients_models = [
        st_gcn.Model(
            in_channels=3,
            num_class=60,
            dropout=0.5,
            edge_importance_weighting=True,
            graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        ).to(dev)
        for _ in range(model_num)
    ]
    server_models = [
        st_gcn.Model(
            in_channels=3,
            num_class=60,
            dropout=0.5,
            edge_importance_weighting=True,
            graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        ).to(dev)
        for _ in range(model_num)
    ]

    # 初始化对应位置的模型有不同权重
    for model in clients_models:
        initialize_weights(model)
    for i in range(model_num):
        assert not model_the_same(clients_models[i], server_models[i])

    # 启动服务器
    server_obj = threading.Thread(
        target=recv_many_model_with_number,
        args=(
            fl_server_addr,
            server_models,
        ),
    )
    server_obj.start()
    # 等待服务器启动
    time.sleep(2)

    # 启动客户端
    clients_objs = []
    for i in range(model_num):
        th_obj = threading.Thread(
            target=send_model_with_number,
            args=(("127.0.0.1", fl_server_addr[1]), clients_models[i], i),
        )
        clients_objs.append(th_obj)
    for i in range(model_num):
        clients_objs[i].start()

    # 等待客户端结束
    for i in range(model_num):
        clients_objs[i].join()
    # 等待服务器结束
    server_obj.join()

    # NOTE 发送与接收顺序不一定相同 需要一个客户端的标志用来排序

    # 确保接收之后对应位置模型相同
    for i in range(model_num):
        assert model_the_same(clients_models[i], server_models[i])


def test_net_with_number():
    import time
    import threading

    th_obj = threading.Thread(
        target=recv_model_with_number,
        args=(
            fl_server_addr,
            model2,
        ),
    )
    th_obj.start()
    time.sleep(2)
    send_model_with_number(("127.0.0.1", fl_server_addr[1]), model, 1)
    th_obj.join()


def all_test():
    set_variables()
    test_assign_model()
    test_speed()
    test_net()
    test_net_with_number()
    test_many_steam()


if __name__ == "__main__":
    # NOTE GPU 模型比 CPU 模型大 在不同设备上能否加载同一个state_dict?
    device_size_diff()
    all_test()
    print("all test passed")
