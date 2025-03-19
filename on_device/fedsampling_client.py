import json
import sys
import socket
import traceback
import pickle
import argparse
import yaml
from rich.console import Console
from rich.padding import Padding
from pathlib import Path
import sys
import io
import struct
import threading
import multiprocessing
from torch.utils.data import DataLoader, Subset
import torch
import queue
import selectors
import copy
import time
import numpy as np
from collections import OrderedDict

PROJECT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
sys.path.append(PROJECT_DIR.joinpath("src").as_posix())
from utls.utils import (
    TRAIN_LOG,
    Logger,
    fix_random_seed,
    NN_state_load,
    get_argparser,
    evaluate
)

from utls.models import MODEL_DICT
from data.utils.datasets import DATA_NUM_CLASSES_DICT, DATASETS , DATASETS_COLLATE_FN
from utls.dataset import CustomSampler
from utls.utils import Timer


console = Console()  # 终端输出对象
client_lock = threading.RLock()  # 多线程的client访问锁
read_finish = threading.Event()
write_finish = threading.Event()
print_lock = multiprocessing.RLock()  # 多进程的输出锁

class Estimator:
    def __init__(self,M,alpha,dataset_len):
        self.M = M # size threshold；为了便于实现，这里设置为所有客户数据量的最大值+1
        self.alpha = alpha # 公式(1)中的alpha
        self.real_response = dataset_len
    
    def query(self):
        fake_response = np.random.randint(1,self.M) #也就是随机取
        choice = np.random.binomial(n=1,p=self.alpha)# n=1 意味着每次只进行一次试验;p=self.alpha: 这是成功的概率
        # 函数返回一个整数，表示在 n 次试验中成功的次数。由于 n=1，返回值将是 0 或 1;0：表示失败。
        response = choice*self.real_response + (1-choice)*fake_response
        return response

class BaseClient:
    def __init__(self, client_id, train_index, batch_size,estimator):
        self.client_id = client_id
        self.train_set_index = np.array(train_index)
        self.train_set_len = len(train_index)
        self.participation_times = 0

        self.batch_size = batch_size

        self.training_time = 0
        self.pretrained_accuracy = 0
        self.accuracy = 0
        self.loss = 0.0
        self.grad = None #存梯度值
        self.buffer = None # 存persistent_buffers

        self.training_time_record = {}
        self.set_estimator(estimator['M'],estimator['alpha'])

    def participate_once(self):
        self.participation_times += 1

    def set_estimator(self , M, alpha):
        self.estimator = Estimator(M,alpha,self.train_set_len)

    def neet_to_send(self):
        return {}

class Trainer:
    def __init__(
            self,
            args
    ):
        self.args = args
        self.device ="cuda"
        self.data_num_classes = DATA_NUM_CLASSES_DICT[self.args['dataset']]
        self.model = MODEL_DICT[self.args["model"]](self.data_num_classes) # 暂时放置在CPU上
        self.current_client_instance = None

        self.trainset = DATASETS[self.args['dataset']](PROJECT_DIR / "data" / self.args["dataset"], "train")
        self.train_sampler = CustomSampler(list(range(len(self.trainset))))
        self.trainloader = DataLoader(Subset(self.trainset, list(range(len(self.trainset)))), self.args["batch_size"], num_workers=2,collate_fn = DATASETS_COLLATE_FN[self.args['dataset']], persistent_workers=True,
                                      sampler=self.train_sampler,)
        self.local_epoch = self.args["local_epoch"]
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)
        self.optimizer = None
        self.timer = Timer()  # 训练计时器
        self.KN = 0
        self.K = 0
    
    def load_dataset(self):
        # 对于每个 candidate_sample，生成一个独立的随机结果，表示在一次试验中成功的概率为 KN
        choosed = np.random.binomial(size=(self.current_client_instance.train_set_len,), n=1, p=min(1.0,self.KN))
        candidate_set_index=np.array(self.current_client_instance.train_set_index)
        participate_set_index = candidate_set_index[choosed == 1].tolist()
        self.current_client_instance.selected_data_num=len(participate_set_index)
        self.trainloader.sampler.set_index(participate_set_index)  # 在里面实现了深拷贝
        self.trainloader.batch_sampler.batch_size = self.current_client_instance.batch_size
    
    def set_parameters(self,model_parameters):
        self.model.load_state_dict(model_parameters)
        self.model = self.model.to(self.device) # 转移到gpu上
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),lr=self.args["lr"],momentum=self.args["momentum"],weight_decay=self.args["weight_decay"],)

    def start(self,global_epoch, client_instance, model_parameters):
        self.timer.start()
        self.current_client_instance = client_instance
        self.set_parameters(model_parameters)  # 设置参数
        self.load_dataset()

        self.local_train() # 本地训练
        self.model = self.model.to("cpu") # 训练完成后放置到CPU上

        # 拷贝模型参数
        delta = OrderedDict()
        for name , param in self.model.named_parameters():
            delta[name] = self.current_client_instance.selected_data_num * (model_parameters[name] - param.data)
        buffer = OrderedDict()
        for name , param in self.model.named_buffers():
            buffer[name] = self.current_client_instance.selected_data_num * copy.deepcopy(param.data)
        current_client_instance_model_dict = {"delta":delta,"buffer":buffer}  # 一定要深拷贝
        self.timer.stop()

        self.current_client_instance.training_time_record[global_epoch] = self.timer.times[-1]
        self.current_client_instance.participate_once()

        # 返回训练后的模型参数，训练时间
        return current_client_instance_model_dict, self.current_client_instance.training_time_record[global_epoch]

    def full_set(self):
        self.model.train()
        for _ in range(self.local_epoch):
            for inputs, targets in self.trainloader:
                if isinstance(inputs,torch.Tensor):
                    inputs = inputs.to(self.device, non_blocking=True)
                else:
                    inputs = [tensor.to(self.device, non_blocking=True) for tensor in inputs]
                targets = targets.to(self.device,non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets).mean() # 两者会有稍许误差
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()
    
    def local_train(self):
        """
        本地训练函数，后面实现的算法要重写这个函数.
        可以确定的是：在训练前，模型已经在设备上了
        """
        self.full_set()




def encode(obj):
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def decode(pickle_bytes, encoding='utf-8'):
    obj = pickle.loads(pickle_bytes, encoding=encoding)
    return obj


def json_encode(obj, encoding):
    return json.dumps(obj, ensure_ascii=False).encode(encoding)


def json_decode(json_bytes, encoding):
    tiow = io.TextIOWrapper(io.BytesIO(json_bytes), encoding=encoding, newline="")
    obj = json.load(tiow)
    tiow.close()
    return obj



class ReadThread(threading.Thread):
    def __init__(self):
        super().__init__()
        global client , client_lock
        self.client = client
        self.client_lock = client_lock
        self._recv_buffer = b""
        self._jsonheader_len = None
        self.jsonheader = None
        self.server_2_client_data = None
        self.finishedRead = False

    def _read(self):
        try:
            data = self.client.socket_manager.sock.recv(20_971_520)  # 20MB
        except BlockingIOError:
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.") # 当对端正常关闭连接（发送FIN包）后，本端调用recv()会返回空字节，表示对端已关闭连接。

    def process_jsonheader(self):
        hdrlen = self._jsonheader_len
        if len(self._recv_buffer) >= hdrlen:
            self.jsonheader = json_decode(
                self._recv_buffer[:hdrlen], "utf-8"
            )
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_protoheader(self):
        hdrlen = 2
        if len(self._recv_buffer) >= hdrlen:
            self._jsonheader_len = struct.unpack(">H", self._recv_buffer[:hdrlen])[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_response(self):
        content_len = self.jsonheader["content-length"]
        if not len(self._recv_buffer) == content_len:  # 还没收到完整数据
            return
        raw_data = self._recv_buffer[:content_len]
        self.server_2_client_data = decode(raw_data)

        server_2_client_time = time.time() - self.server_2_client_data['timestamp']
        self.server_2_client_data = self.server_2_client_data['content']
        self.server_2_client_data["server_2_client_time"] = server_2_client_time
        with self.client_lock:
            self.client.received_data = self.server_2_client_data
        self.finishedRead = True

    def run(self):
        while True:
            self._read()
            if self._jsonheader_len is None:
                self.process_protoheader()
            if self._jsonheader_len is not None:
                if self.jsonheader is None:
                    self.process_jsonheader()
            if self.jsonheader:
                if not self.finishedRead:
                    self.process_response()
                else:
                    break


class MyThread(threading.Thread):  # 每个线程与一个进程对应
    def __init__(self,):
        super().__init__()
        global client , client_lock, print_lock
        self.client = client
        self.client_lock = client_lock
        self.print_lock = print_lock
        self.daemon = True  # 设置为守护进程

    def run(self):
        while True:
            need_to_send = self.client.need_to_send_queue.get()
            write_process = WriteProcess(self.client.server_ip_port, need_to_send , self.print_lock)
            write_process.start()
            write_process.join()
            with self.client_lock:
                self.client.need_to_send_num -= 1
                if self.client.need_to_send_num == 0:
                    write_finish.set()

            


class WriteProcess(multiprocessing.Process):
    def __init__(self, server_ip_port, need_to_send, print_lock):
        super().__init__()
        self.need_to_send = need_to_send
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)
        self.sock.connect_ex(server_ip_port) # 创建一个连接
        self.print_lock = print_lock
        self._send_buffer = b""  # 写缓冲区

    def _create_message(
            self, content
    ):
        jsonheader = {
            "content-length": len(content),
        }
        jsonheader_bytes = json_encode(jsonheader, "utf-8")
        message_hdr = struct.pack(">H", len(jsonheader_bytes))
        message = message_hdr + jsonheader_bytes + content
        return message

    def _create_response(self):
        response = dict(timestamp = time.time(), content = self.need_to_send)
        response = encode(response)
        return response

    def run(self):
        with self.print_lock:
            console.log(f"start sending to server")
        response = self._create_response()
        message = self._create_message(response)  # 加两个头文件
        self._send_buffer += message
        while True:
            if self._send_buffer:
                try:
                    sent = self.sock.send(self._send_buffer)
                except BlockingIOError:
                    pass
                else:
                    self._send_buffer = self._send_buffer[sent:]
            else:
                break
        with self.print_lock:
            console.log(f"send to server successfully")

        try:
            self.sock.close()
        except OSError as e:
            with self.print_lock:
                console.log(f"Error: socket.close() exception: {e!r}")
        finally:
            self.sock = None

    


class clientsocket:
    def __init__(self, server_ip, server_port, name):
        self.server_ip = server_ip
        self.server_port = server_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)
        self.sock.connect_ex((self.server_ip,self.server_port)) # 创建一个连接
        self.need_to_send = {"name": name,"action":"register"}
    
    def _create_message(
            self, content
    ):
        jsonheader = {
            "content-length": len(content),
        }
        jsonheader_bytes = json_encode(jsonheader, "utf-8")
        message_hdr = struct.pack(">H", len(jsonheader_bytes))
        message = message_hdr + jsonheader_bytes + content
        return message

    def _create_response(self):
        response = dict(timestamp = time.time(), content = self.need_to_send)
        response = encode(response)
        return response

    def send(self):
        console.log(f"start registering to server")
        response = self._create_response()
        message = self._create_message(response)  # 加两个头文件
        _send_buffer = b""
        _send_buffer += message
        while True:
            if _send_buffer:
                try:
                    sent = self.sock.send(_send_buffer)
                except BlockingIOError:
                    pass
                else:
                    _send_buffer = _send_buffer[sent:]
            else:
                break
        console.log(f"send to server successfully")

    def close(self):
        print(f"Closing connection to {self.server_ip}")
        try:
            self.sock.close()
        except OSError as e:
            print(f"Error: socket.close() exception for {self.addr}: {e!r}")
        finally:
            # Delete reference to socket object for garbage collection
            self.sock = None


class Client:
    def __init__(self, args, socket_manager, name):
        self.args = args
        self.socket_manager = socket_manager
        self.server_ip_port = (self.socket_manager.server_ip , self.socket_manager.server_port)
        self.name = name  # 设备名称
        self.received_data = None

        self.need_to_send_queue = queue.Queue() # 给写守护进程的发送队列
        self.need_to_send_num = 0 # 标志是否都发送完成，不可由need_to_send_queue.qsize()代替。因为qsize()==0时，可能还在发送

        self.client_ids = [] # 存储本地要训练的client的id号
        self.clientId_dataIndex = {} # client的id号和其训练数据的索引

        self.client_instances_dict = {} # client的id号和其对应的client实例

        self.trainer = Trainer(self.args) # 训练器

        self.current_epoch_transmission = 0 # 当前轮次的server发送给client的传输时间

        self.current_selected_client_ids = [] # 当前全局训练轮次中被选中的client

        self.model = None # 记录本轮次中从server接收的全局模型参数

def read_from_server():
    read_thread = ReadThread()
    read_thread.start()
    read_thread.join()



def run():
    global client, client_lock
    # ============== register================
    client.socket_manager.send()
    read_from_server()
    client.client_ids = client.received_data["clients"]
    client.clientId_dataIndex = client.received_data["data_indices"]

    console.log(f"device {client.name} need to train {client.client_ids}" , style='red')
    # 实例化本地的client
    for client_id in client.client_ids:
        client.client_instances_dict[client_id] = BaseClient(client_id, client.clientId_dataIndex[client_id], client.args["batch_size"],client.received_data['estimator'])
    
    console.log("clients has been initialized successfully")


    client_2_server_data = dict(name=client.name,action= "check") 
    with client_lock:
        client.need_to_send_num += 1
        write_finish.clear() 
    client.need_to_send_queue.put(client_2_server_data)
    write_finish.wait()
    write_finish.clear() 
    # while True:
    #     with client_lock:
    #         if client.need_to_send_num == 0:
    #             break # 全部发送完成，一轮全局结束
    #     time.sleep(1) 
    
    # =========== 开始训练 =======================
    while True:
        # ======== 接收 ==============
        read_from_server() 



        if client.received_data['finished']:
            break
        console.rule(f"start {client.received_data['global_epoch']}",style='red')
        console.log(f"transmission time is {client.received_data['server_2_client_time']}")
        client.model = client.received_data['model']
        client.current_epoch_transmission = client.received_data['server_2_client_time']
        client.current_selected_client_ids = client.received_data['current_selected_client_ids']
        client.trainer.KN = client.received_data['KN']
        client.trainer.K = client.received_data['K']
        console.log(f"client.trainer.KN: {client.trainer.KN}, client.trainer.K: {client.trainer.K}")
        console.log(f"need to train {client.current_selected_client_ids} in global epoch {client.received_data['global_epoch']}")
        for client_id in client.current_selected_client_ids:
            assert client_id in client.client_ids , f"{client_id} do not belongs to the device" 
            current_client_instance_model_dict, training_time = client.trainer.start(client.received_data['global_epoch'],client.client_instances_dict[client_id],client.model)
            
            client_2_server_data = dict(
                                        name=client.name, # 设备名称
                                        action="upload", # 行为
                                        client_id=client_id,  # 训练的client id号
                                        client_model = current_client_instance_model_dict, # 模型参数
                                        weight = client.client_instances_dict[client_id].train_set_len, # 权重
                                        s2c_training_time = training_time + client.current_epoch_transmission, # 下发与训练的时间
                                        **client.client_instances_dict[client_id].neet_to_send())  # 客户需要发送的信息
            console.log(f"{client_id} has finished training, using {training_time}s")
            with client_lock:
                client.need_to_send_num += 1
                write_finish.clear()
            client.need_to_send_queue.put(client_2_server_data)

        console.log(f"{client.name} has finished local training of all selected clients")

        write_finish.wait()
        write_finish.clear()
        # while True:
        #     with client_lock:
        #         if client.need_to_send_num == 0:
        #             break # 全部发送完成，一轮全局结束
        #     time.sleep(1) 
        # =============================





if __name__ == '__main__':
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    
    socket_manager = clientsocket(args['server_ip'], args['server_port'], parser.name)
    client = Client(args, socket_manager, parser.name)
    write_daemon_thread = MyThread()
    write_daemon_thread.start()
    
    run()

    client.socket_manager.close()
