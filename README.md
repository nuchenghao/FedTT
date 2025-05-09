Repertory for *It Takes Two: Accelerating Accurate Federated Learning through Pipelined Intra-Batch Data Sampling and Training*


# The hardware configuration of the workstation is as follows:
+ CPU: AMD Ryzen 9 7950X 16-Core Processor
+ GPU: NVIDIA 4090 * 2
+ Memory: 128GB DDR5 5600MHz

# The software configuration of the workstation is as follows:
+ System: Ubuntu 20.04.6 LTS
+ NVIDIA Driver Version: 550.135
+ CUDA Driver Version: 12.1
+ Pytorch: 2.5.1
+ Python3: 3.12.7

# Run FedTT

## Simulations

+ The FL training framework consists of one server, one trainer, and multiple clients. The clients store client-specific attributes, such as data indices and local model parameters. In each global epoch, the trainer sequentially performs the local training for the selected clients. The duration of each global epoch is determined by the slowest client.

+ All simulation startup commands are stored in `start.sh`
+ **Before running all FL experiments, make sure to execute the data generation commands. These commands are also included in the script file.**

### cifar-100
The dataset is stored in the `FedTT/data/cifar100` directory.

### cinic-10
For the cinic-10 dataset, we merge its `train` and `valid` sets to obtain a larger training set, which is then partitioned across different clients.

1. Put the `CINIC-10.tar.gz` file into the `FedTT/data/cinic10/raw` directory.
2. Execute the following commands:

```bash
cd ./data/cinic10/raw
tar -zxvf CINIC-10.tar.gz
cd ../../utils
python3 split_cinic10.py
```

### snli
1. Download the [SNLI 1.0](https://nlp.stanford.edu/projects/snli/) and [glove.840B.300d.zip](https://nlp.stanford.edu/projects/glove/) to the `FedTT/data/snli` directory.
2. Execute the following commands:

```bash
cd ./data/snli
unzip snli.zip -d . && mv ./snli_1.0/* . && rm -rf snli_1.0
cd ../utils
python3 snli.py
```

### domainnet
Download these six [sub-datasets](https://ai.bu.edu/M3SDA/) to the `FedTT/data/domainnet`, unzip these, and put the `.txt`s to the `FedTT/data/domainnet/splits`.


## Testbed experiments
+ All testbed experiment startup commands are stored in `start_on_device.sh`
+ You need to change the server IP address and port in the algorithm configuration file under the `FedTT/on_device/config` directory.