# fedavg + st_gcn

### 运行方式：

1. 创建一个虚拟环境，这里用的是python3.8，按照[GitHub - wanjinchang/st-gcn: Spatial Temporal Graph Convolutional Networks (ST-GCN) for Skeleton-Based Action Recognition in PyTorch](https://github.com/wanjinchang/st-gcn) 的requirements，配置st-gcn所需环境

   ```
   git clone https://github.com/yysijie/st-gcn.git
   cd st-gcn
   pip install -r requirements.txt
   ```

   接着自行配置torch和GPU

2. 克隆本仓库代码：
   ```
   git clone https://github.com/Duanice/FedAvg-ST-GCN.git
   ```
   数据集按照下图组织：
   ![数据集组织结构](resource/数据集组织结构.png)

4. 联邦训练：对于Kinetics数据集需要重新配置Server和Client的参数，参考st-gcn源码config文件夹下的yaml文件修改即可。在fl_st目录运行：

   ```
   python Server.py
   ```

### 通信模型:
      
      NOTE:分成两个循环：
      
      客户端 listen -> 接收模型 -> 训练一个epoch -> 发起通信 -> 上传模型 -> (重复)listen
      
      服务器 发起通信 -> 下放模型 -> listen -> 接收模型 -> 聚合 -> (重复)发起通信

