import json
import socket
import selectors
import random
import logging
import copy
import threading
import time
import queue
from rich.console import Console
from rich.padding import Padding
import swanlab
import os
import pickle
import multiprocessing
import torch
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
import yaml
import io
import struct
from pathlib import Path
import numpy as np
import sys
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
from typing import Dict, List
from utls.models import MODEL_DICT
from data.utils.datasets import DATA_NUM_CLASSES_DICT, DATASETS , DATASETS_COLLATE_FN
from utls.dataset import CustomSampler


console = Console()  # 终端输出对象
server_lock = threading.RLock()  # 多线程的stateInServer锁
write_finish = threading.Event()
read_finish = threading.Event() 
print_lock = multiprocessing.RLock()  # 多进程的输出锁



class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:0'
        self.current_time = 0  # 全局时间
        

        self.client_sample_stream = [
            random.sample(
                list(range(self.args["client_num"])), max(1, int(self.args["client_num"] * self.args["client_join_ratio"]))
            )
            for _ in range(self.args["global_epoch"])
        ]
        self.current_selected_client_ids: List[int] = []

        self.data_num_classes = DATA_NUM_CLASSES_DICT[self.args['dataset']]
        self.model = MODEL_DICT[self.args["model"]](self.data_num_classes)  # 先放置在CPU上 

        self.testset = DATASETS[self.args['dataset']](PROJECT_DIR / "data" / args["dataset"], "test")
        self.testloader = DataLoader(Subset(self.testset, list(range(len(self.testset)))), batch_size=self.args['t_batch_size'],
                                     shuffle=False, pin_memory=True, num_workers=4,collate_fn = DATASETS_COLLATE_FN[self.args['dataset']],
                                     persistent_workers=True, pin_memory_device=self.device,prefetch_factor = 8)
        
        self.client_model_cache = queue.Queue()
        self.weight_cache = queue.Queue()
        self.metadata = queue.Queue()


        self.current_global_epoch = 0 # 已完成的次数

        self.ltr = 0.0
        self.ddlr = 1.0 
        self.ddl_R = 0.0
        self.lt = 0.0
        self.U = []
        self.w = 5
        self.lss =0.05
        self.dss = 0.05

    def select_clients(self, global_epoch):
        self.current_selected_client_ids = self.client_sample_stream[global_epoch - 1] #我们的全局从1开始


    def get_model_dict(self):
        return {key: value for key, value in self.model.state_dict().items()}

    def lt_selection_next_round(self,LLow,LHigh):
        #  𝑙𝑡 selection for next (𝑅 + 1)-th round  (Algorithm 2)
        ll = min(LLow)
        lh = sum(LHigh) / len(LHigh)
        self.lt = ll + (lh - ll) * self.ltr
    
    def ltr_ddlr_control(self,LSum_R,L_R):
        self.U.append( LSum_R / (L_R * self.ddl_R))
        if self.current_global_epoch % self.w == 0 :
            if len(self.U) >= 2 * self.w and sum(self.U[self.current_global_epoch - 2 * self.w:self.current_global_epoch - self.w]) > sum(self.U[self.current_global_epoch - self.w:]):
                self.ltr = min(self.ltr + self.lss , 1.0)
                self.ddlr = max(self.ddlr - self.dss,0.0) 
            else:
                self.ltr = max(self.ltr - self.lss , 0.0)
                self.ddlr = min(self.ddlr + self.dss , 1.0)
    
    def select_deadline(self,E, clientId_info):
        def find_peak_ddl_E(epoch):
            completeTime = []
            ddl_E = []
            t = 10.0
            for client_id in self.current_selected_client_ids:
                completeTime.append(
                    (clientId_info[client_id]["len_OT"] + clientId_info[client_id]["batch_size"] - 1 ) / clientId_info[client_id]["batch_size"] * np.array(clientId_info[client_id]["batch_training_time"]).mean() * epoch 
                    + np.array(clientId_info[client_id]["network"]).mean()
                )
            completeTime =  np.array(completeTime)
            count = 0
            while count != len(self.current_selected_client_ids):
                count = np.sum(completeTime < t)
                ddl_E.append((count / t , t))
                t += 1.0
            return max(ddl_E, key=lambda x: (x[0], x[1]))[1]
        dl = find_peak_ddl_E(1)
        dh = find_peak_ddl_E(E)
        self.ddl_R = dl + (dh - dl) * self.ddlr
        console.log(f"deadline: {self.ddl_R}, loss_threshold: {self.lt}, ddlr: {self.ddlr}, ltr: {self.ltr}")

    def aggregate(self):
        with torch.no_grad():
            client_model_cache = []
            while True:
                try:
                    element = self.client_model_cache.get_nowait()
                    client_model_cache.append(element)
                except queue.Empty:
                    break
            weight_cache = []
            while True:
                try:
                    element = self.weight_cache.get_nowait()
                    weight_cache.append(element)
                except queue.Empty:
                    break
            weights = torch.tensor(weight_cache) / sum(weight_cache)
            model_list = [list(delta.values()) for delta in client_model_cache]
            aggregated_model = [
                torch.sum(weights * torch.stack(grad, dim=-1), dim=-1)
                for grad in zip(*model_list)
            ]
            averaged_state_dict = OrderedDict(zip(client_model_cache[0].keys(), aggregated_model))
            self.model.load_state_dict(averaged_state_dict)

        LLow = []
        LHigh = []
        Lsum = []
        while True:
            try:
                metadata = self.metadata.get_nowait()
                LLow.append(metadata['llow'])
                LHigh.append(metadata['lhigh'])
                Lsum.append(metadata['lsum'])
            except queue.Empty:
                break
        L_R = sum(weight_cache)
        self.current_global_epoch += 1 
        self.lt_selection_next_round(LLow,LHigh)
        self.ltr_ddlr_control(sum(Lsum),L_R)
    
    def neet_to_send(self):
        return {"synchronization":{"deadline":self.ddl_R, "loss_threshold" : self.lt}}

        



def encode(obj):
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def decode(pickle_bytes, encoding='utf-8'):
    obj = pickle.loads(pickle_bytes, encoding=encoding)
    return obj


def json_encode(obj, encoding):
    return json.dumps(obj, ensure_ascii=False).encode(encoding)


def json_decode(json_bytes, encoding):
    tiow = io.TextIOWrapper(
        io.BytesIO(json_bytes), encoding=encoding, newline=""
    )
    obj = json.load(tiow)
    tiow.close()
    return obj


class ReadThread(threading.Thread):  # 每个线程与一个进程对应
    def __init__(self, physical_device, keep):
        super().__init__()
        global server, server_lock, print_lock
        self.server = server
        self.server_lock = server_lock
        self.print_lock = print_lock
        self.physical_device = physical_device
        self.keep = keep

    def run(self):
        with multiprocessing.Manager() as manager:
            multiprocessing_shared_queue = manager.Queue()
            read_process = ReadProcess(self.physical_device, self.print_lock,multiprocessing_shared_queue, self.keep)
            read_process.start()
            read_process.join()

            option, physical_device_id, client_2_server_data = multiprocessing_shared_queue.get()
        
        if option == "register":  # 一个注册进程
            with self.server_lock:
                physical_device = self.server.all_physical_device_queue[physical_device_id]
                assert physical_device.physical_device_id == physical_device_id, f"the physical device id {physical_device_id} is not equal to physical_device.physical_device_id {physical_device.physical_device_id}"
                physical_device.name = client_2_server_data.get("name") # 在子线程中进行修改，不能在子进程中修改
                self.server.wait_queue.put("register")
                if self.server.wait_queue.qsize() == self.server.need_connect_device:
                    self.server.current_state = 'registered'
                    self.server.wait_queue.queue.clear()

        elif option == "check":
            assert physical_device_id == -1
            with self.server_lock:
                self.server.wait_queue.put("check")
                if self.server.wait_queue.qsize() == self.server.need_connect_device:
                    self.server.current_state = "checked"
                    self.server.wait_queue.queue.clear()
        
        elif option == 'test':
            assert physical_device_id == -1
            with self.server_lock:
                self.server.clientId_info[client_2_server_data["client_id"]]["len_OT"] = client_2_server_data['info']["len_OT"]
                self.server.clientId_info[client_2_server_data["client_id"]]['batch_size'] = client_2_server_data["info"]['batch_size'] 
                self.server.clientId_info[client_2_server_data["client_id"]]['batch_training_time'].append(client_2_server_data['info']['batch_training_time'])
                self.server.clientId_info[client_2_server_data["client_id"]]['network'].append(client_2_server_data['s2c_time'])

                self.server.wait_queue.put("test")
                if self.server.wait_queue.qsize() == self.server.need_to_uploaded:
                    self.server.current_state = 'tested'
                    self.server.wait_queue.queue.clear()
                
        
        elif option == 'upload':
            assert physical_device_id == -1
            with self.server_lock:
                self.server.trainer.client_model_cache.put(client_2_server_data['client_model']) # 放置模型参数
                self.server.trainer.weight_cache.put(client_2_server_data['weight']) # 放置权重
                self.server.trainer.metadata.put(client_2_server_data['metadata'])
                self.server.clientId_info[client_2_server_data["client_id"]]["len_OT"] = client_2_server_data['info']["len_OT"]
                self.server.clientId_info[client_2_server_data["client_id"]]['batch_size'] = client_2_server_data["info"]['batch_size'] 
                self.server.clientId_info[client_2_server_data["client_id"]]['batch_training_time'].append(client_2_server_data['info']['batch_training_time'])
                self.server.clientId_info[client_2_server_data["client_id"]]['network'].append(client_2_server_data['s2c_time'])
                self.server.globel_epoch_training_time[self.server.current_epoch].append((client_2_server_data['client_id'],client_2_server_data['s2c_training_time'] + client_2_server_data["client_2_server_time"])) # 记录训练时间
                self.server.wait_queue.put('upload')
                if self.server.wait_queue.qsize() == self.server.need_to_uploaded:
                    self.server.current_state = 'uploaded'
                    self.server.wait_queue.queue.clear()


class ReadProcess(multiprocessing.Process):
    def __init__(self, physical_device, print_lock, multiprocessing_shared_queue , keep):
        super().__init__()
        self.physical_device = physical_device
        self.print_lock = print_lock
        self.multiprocessing_shared_queue = multiprocessing_shared_queue  # 与父进程的某个子线程进行信息传递的队列
        self._recv_buffer = b""  # 接收缓冲区
        self._jsonheader_len = None
        self.jsonheader = None
        self.client_2_server_data = None
        self.keep = keep

    def _read(self):
        try:
            data = self.physical_device.sock.recv(20_971_520)  # read 20MB once
        except BlockingIOError:
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                assert data == b""
                if not self.keep:
                    try:
                        self.physical_device.sock.close()
                    except OSError as e:
                        console.log(f"Error: socket.close() exception for {self.physical_device.address}: {e!r}")
                    finally:
                        self.physical_device.sock = None
                    # raise RuntimeError("Peer closed.")

    def process_protoheader(self):
        hdrlen = 2
        if len(self._recv_buffer) >= hdrlen:
            self._jsonheader_len = struct.unpack(">H", self._recv_buffer[:hdrlen])[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_jsonheader(self):
        hdrlen = self._jsonheader_len
        if len(self._recv_buffer) >= hdrlen:
            self.jsonheader = json_decode(self._recv_buffer[:hdrlen], "utf-8")
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_request(self):
        content_len = self.jsonheader["content-length"]
        if not len(self._recv_buffer) == content_len:
            return
        # 全部数据均已接收
        raw_data = self._recv_buffer[:content_len]
        self.client_2_server_data = decode(raw_data)

        client_2_server_time = time.time() - self.client_2_server_data["timestamp"]
        self.client_2_server_data = self.client_2_server_data["content"]
        self.client_2_server_data["client_2_server_time"] = client_2_server_time

        if self.client_2_server_data.get('action') == 'register': 
            name = self.client_2_server_data.get('name')
            with self.print_lock:
                console.log(f"Received {name} register request and transmission time is {client_2_server_time}", style="bold yellow")
            self.multiprocessing_shared_queue.put(("register", self.physical_device.physical_device_id, self.client_2_server_data))  # 返回给server修改,第2个参数表示对应的message


        elif self.client_2_server_data.get('action') == 'check':
            name = self.client_2_server_data.get('name')
            with self.print_lock:
                console.log(f"Received {name} check info and transmission time is {client_2_server_time}", style="bold yellow")
            self.multiprocessing_shared_queue.put(('check', self.physical_device.physical_device_id, self.client_2_server_data))
        
        elif self.client_2_server_data.get("action") == 'test':
            name = self.client_2_server_data.get('name')
            with self.print_lock:
                console.log(f"Received {name}'s {self.client_2_server_data.get("client_id")} test info and transmission time is {client_2_server_time}", style="bold yellow")
            self.multiprocessing_shared_queue.put(('test', self.physical_device.physical_device_id, self.client_2_server_data))

        elif self.client_2_server_data.get('action') == 'upload':
            name = self.client_2_server_data.get('name')
            client_id = self.client_2_server_data.get('client_id')
            with self.print_lock:
                console.log(f"Received {name}'s {client_id} upload info and transmission time is {client_2_server_time}", style="bold yellow")
            self.multiprocessing_shared_queue.put(('upload', self.physical_device.physical_device_id, self.client_2_server_data))

    def run(self):
        while True:
            self._read()
            if self._jsonheader_len is None:
                self.process_protoheader()
            if self._jsonheader_len is not None:
                if self.jsonheader is None:
                    self.process_jsonheader()
            if self.jsonheader:
                if self.client_2_server_data is None:
                    self.process_request()
                elif self.client_2_server_data is not None: # 这里elif会让_read()再执行一次，从而关闭对向socket
                    break
        if not self.keep and self.physical_device.sock != None:
            try:
                self.physical_device.sock.close()
            except OSError as e:
                console.log(f"Error: socket.close() exception for {self.physical_device.address}: {e!r}")
            finally:
                self.physical_device.sock = None

class WriteThread(threading.Thread):  # 每个线程与一个进程对应
    def __init__(self,content_server_2_client, physical_device):
        super().__init__()
        global server, server_lock, print_lock
        self.server = server
        self.server_lock = server_lock
        self.print_lock = print_lock
        self.physical_device = physical_device
        self.content_server_2_client = content_server_2_client
        self.multiprocessing_shared_queue = multiprocessing.Queue()

    def run(self):
        write_process = WriteProcess(self.content_server_2_client, self.physical_device, self.print_lock, self.multiprocessing_shared_queue)
        write_process.start()
        write_process.join()

        option, physical_device_id = self.multiprocessing_shared_queue.get()
        if option == "distribute":
            with self.server_lock:
                physical_device = server.all_physical_device_queue[physical_device_id]
                assert physical_device.physical_device_id == physical_device_id, f"the physical device id {physical_device_id} is not equal to physical_device.physical_device_id {physical_device.physical_device_id}"
                server.wait_queue.put("distribute")
                if server.wait_queue.qsize() == server.need_distribute_device:
                    server.current_state = 'distributed' 
                    server.wait_queue.queue.clear()
                    write_finish.set()

class WriteProcess(multiprocessing.Process):
    def __init__(self, content_server_2_client, physical_device, print_lock, multiprocessing_shared_queue):
        super().__init__()
        self.content_server_2_client = content_server_2_client
        self.physical_device = physical_device
        self.print_lock = print_lock
        self.multiprocessing_shared_queue = multiprocessing_shared_queue  # 共享队列
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
        response = dict(timestamp = time.time(), content = self.content_server_2_client)
        response = encode(response)
        return response

    def run(self):
        with self.print_lock:
            console.log(f"start sending to {self.physical_device.name} whose physical device id is {self.physical_device.physical_device_id}")
        response = self._create_response()
        message = self._create_message(response)  # 加两个头文件
        self._send_buffer += message
        while True:
            if self._send_buffer:
                try:
                    sent = self.physical_device.sock.send(self._send_buffer)
                except BlockingIOError:
                    pass
                else:
                    self._send_buffer = self._send_buffer[sent:]
            else:
                break
        self.multiprocessing_shared_queue.put(("distribute", self.physical_device.physical_device_id))
        with self.print_lock:
            console.log(f"Sent to {self.physical_device.name} whose physical device id is {self.physical_device.physical_device_id}")




class Server:
    def __init__(self,args, socket_manager):
        self.args = args
        self.global_time = 0 # 记录全局时间 
        self.need_connect_device = self.args["need_connect_device"] 
        self.need_distribute_device = 0
        self.current_device_nums = 0
        self.tot_client_nums = self.args["client_num"]
        self.client_ids = list(range(self.tot_client_nums)) # 所有客户的client_id
        self.client_ids_device = [-1 for _ in range(self.tot_client_nums)]
        self.device_client_ids = {_ : [] for _ in range(self.need_connect_device)}
        
        self.total_epoches = self.args["global_epoch"]

        self.socket_manager = socket_manager

        self.all_physical_device_queue = [] # 记录所有下发物理设备的队列

        self.current_state = None # registered/distributed/checked/received/tested

        self.wait_queue = queue.Queue()  # 多线程共享队列， 存储线程的处理结果; 要注意清空

        self.trainer = Trainer(self.args)

        self.need_to_uploaded = 0

        self.current_epoch = 0
        self.globel_epoch_training_time = {global_epoch:[] for global_epoch in range(1 , 1 + self.total_epoches)} # {global epoch : (client_id , 整轮需要的训练时间)}

        self.clientId_info = { client_id : {"len_OT":0,"batch_size":0,"batch_training_time":[],"network":[]} for client_id in range(self.tot_client_nums)} 

        

        # wandb
        if self.args['wandb']:
            log_dir = f"{PROJECT_DIR}/WANDB_LOG_DIR"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            self.experiment = swanlab.init(
                project=f"{self.args['project']}",
                experiment_name=self.args["experiment_name"],
                config=self.args,
                dir=log_dir,
                reinit=True,
            )
            self.experiment.log({"acc": 0.0}, step=0)
            
    
    
    def add_device(self):
        self.current_device_nums += 1
    
    def close_all_sockets(self):
        for physical_device in self.all_physical_device_queue:
            self.socket_manager.sel.unregister(physical_device.sock)
            physical_device.close()
        self.socket_manager.sel.unregister(self.socket_manager.lsock)
        self.socket_manager.sel.close()


    



class physicalDevice:
    def __init__(self, sock, physical_device_id, addr):
        self.sock = sock
        self.physical_device_id = physical_device_id # socket的id，也对应着设备的id
        self.name = ""
        self.address = addr
        self.client_ids = []
    
    def __repr__(self):  # 日志输出用
        return f"\n{self.name}'s message id is {self.physical_device_id}"  # 为了适应日志输出，加上换行符
    
    def close(self):
        console.log(f"Closing connection to {self.address}")
        try:
            self.sock.close()
        except OSError as e:
            print(f"Error: socket.close() exception for {self.address}: {e!r}")
        finally:
            self.sock = None



class serversocket():
    def __init__(self,server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.sel = selectors.DefaultSelector()
        self.lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Avoid bind() exception: OSError: [Errno 48] Address already in use
        self.lsock.bind((self.server_ip, self.server_port))
        self.lsock.listen()
        console.log(f"Listening on {(self.server_ip, self.server_port)}", style="bold white on blue")
        self.lsock.setblocking(False) # configure the socket in non-blocking mode
        self.sel.register(self.lsock, selectors.EVENT_READ, data=None)

    
    def accept_wrapper(self, sock, physical_device_id = -1):
        conn, addr = sock.accept()
        conn.setblocking(False)
        physical_device = physicalDevice(conn, physical_device_id, addr)
        self.sel.register(conn, selectors.EVENT_READ, data=physical_device)
        return physical_device


def split_list_to_clients(lst, num_clients):
    # 打乱列表以确保随机分配
    shuffled = copy.deepcopy(lst)
    random.shuffle(shuffled)
  
    # 计算每个 client 的基本元素数量和余数
    total = len(shuffled)
    base = total // num_clients
    remainder = total % num_clients
  
    divided = []
    start = 0
  
    # 分配前 num_clients-1 个 client，每个分 base 个元素
    for i in range(num_clients - 1):
        end = start + base
        divided.append(shuffled[start:end])
        start = end
  
    # 最后一个 client 分 base + remainder 个元素
    divided.append(shuffled[start:])
  
    return divided

def split_numbers():
    # 生成0-29的列表
    numbers = list(range(30))
  
    # 打乱顺序
    random.shuffle(numbers)
  
    # 分割列表
    return [
        numbers[:12],      # 第一组12个
        numbers[12:24],    # 第二组12个
        numbers[24:]       # 第三组6个
    ]

def registerStage():
    global server
    # 开始注册
    while True:
        events = server.socket_manager.sel.select(timeout=0)  # 非阻塞调用,立即返回可用的文件描述符,而不等待
        for key, mask in events:
            if key.data is None:  # 服务端的listening socket；意味着有一个新的连接到来，需要注册
                physical_device:physicalDevice = server.socket_manager.accept_wrapper(key.fileobj, physical_device_id = server.current_device_nums)
                server.add_device()
                server.all_physical_device_queue.append(physical_device)
            else:
                physical_device: physicalDevice = key.data  # 获得该物理设备对应的physicalDevice
                server.socket_manager.sel.modify(physical_device.sock, 0, data=physical_device)  # 将这个设备的sock在sel中的状态暂时挂起
                if mask & selectors.EVENT_READ:  # 注册阶段只有读事件
                    read_thread = ReadThread(physical_device,True)
                    read_thread.start()

        with server_lock: # 检查是否注册完成
            if server.current_state == "registered":
                console.log(f"all the devices has registered! The server all_physical_device_queue is {server.all_physical_device_queue}" ,style="red")
                break  # 所有设备都已经注册了
    
    
    # 下发数据集的划分与每个设备上的client的分配
    # client_splits = split_list_to_clients(server.client_ids, server.need_connect_device)
    client_splits = split_numbers()
    for i in range(server.need_connect_device):
        console.log(f"{client_splits[i]}")
        server.all_physical_device_queue[i].client_ids = client_splits[i]
        server.device_client_ids[i] = client_splits[i]
        for client_id in client_splits[i]:
            server.client_ids_device[client_id] = i

    partition_path = PROJECT_DIR / "data" / server.args["dataset"] / "partition.pkl"
    with open(partition_path, "rb") as f:
        partition = pickle.load(f)
    with server_lock:
        server.need_distribute_device = server.need_connect_device
    for physical_device in server.all_physical_device_queue:
        server_2_client_data = {"clients": physical_device.client_ids,
                                "data_indices":{client_id:partition["data_indices"][client_id].tolist() for client_id in physical_device.client_ids},}
        write_thread = WriteThread(server_2_client_data, physical_device)
        write_thread.start()

    write_finish.wait()
    write_finish.clear()
    # while True:
    #     with server_lock:
    #         if server.current_state == "distributed":
    #             console.log("distributed to all clients",style="red")
    #             break # 全部发送完成
    #     time.sleep(1)
    
    # 准备接收check信息
    while True:
        events = server.socket_manager.sel.select(timeout=0)  # 非阻塞调用,立即返回可用的文件描述符,而不等待
        for key, mask in events:
            if key.data is None:  # 服务端的listening socket；意味着有一个新的连接到来，需要注册
                server.socket_manager.accept_wrapper(key.fileobj)
            else:
                physical_device: physicalDevice = key.data  # 获得该物理设备对应的physicalDevice
                server.socket_manager.sel.unregister(physical_device.sock) # 只需要读一次即可
                if mask & selectors.EVENT_READ:  # 注册阶段只有读事件
                    read_thread = ReadThread(physical_device,False)
                    read_thread.start()

        with server_lock:
            if server.current_state == "checked":
                console.log("received all checked info", style="red")
                break  # 接收完成
    


def trainingstage():
    global server

    with server_lock:
        server.need_distribute_device = server.need_connect_device
    # 测试每个client完成的时间
    server.need_to_uploaded = server.tot_client_nums
    for physical_device in server.all_physical_device_queue:
        server_2_client_data = {"model": server.trainer.get_model_dict(),
                                "finished": False,
                                "current_selected_client_ids":physical_device.client_ids, # 测试该设备上的所有客户
                                "global_epoch": 0,
                                **server.trainer.neet_to_send()}
        write_thread = WriteThread(server_2_client_data, physical_device) # 注意要深拷贝
        write_thread.start()
    
    write_finish.wait()
    write_finish.clear()   
    # while True:
    #     with server_lock:
    #         if server.current_state == "distributed":
    #             console.log("distributed to all clients",style="red")
    #             break # 全部发送完成
    #     time.sleep(1)
            
    # 准备接收test信息
    while True:
        events = server.socket_manager.sel.select(timeout=0)  # 非阻塞调用,立即返回可用的文件描述符,而不等待
        for key, mask in events:
            if key.data is None:  # 服务端的listening socket；意味着有一个新的连接到来，需要注册
                server.socket_manager.accept_wrapper(key.fileobj)
            else:
                physical_device: physicalDevice = key.data  # 获得该物理设备对应的physicalDevice
                server.socket_manager.sel.unregister(physical_device.sock) # 只需要读一次即可
                if mask & selectors.EVENT_READ:  # 注册阶段只有读事件
                    read_thread = ReadThread(physical_device,False)
                    read_thread.start()

        with server_lock:
            if server.current_state == "tested":
                console.log("received all the tested of current selected clients", style="red")
                break  # 接收完成






    for global_epoch in range(1,server.total_epoches + 1):
        console.rule(f"start global epoch {global_epoch} training ",style="red")
        server.current_epoch = global_epoch  # 设置当前的轮数
        server.trainer.select_clients(global_epoch) # 设置本轮被选中的客户
        server.need_to_uploaded = len(server.trainer.current_selected_client_ids)
        server.trainer.select_deadline(server.args["local_epoch"], server.clientId_info)


        device_current_selected_client_ids = {device_id:[] for device_id in range(server.need_connect_device)} #{设备号：[被选中的id]}
        for current_selected_client_id in server.trainer.current_selected_client_ids: # 遍历被选择的客户id，将device_current_selected_client_ids进行统计
            device_current_selected_client_ids[server.client_ids_device[current_selected_client_id]].append(current_selected_client_id)
        console.log(f"current selected client ids is {device_current_selected_client_ids}")



        with server_lock:
            server.need_distribute_device = 0
            for physical_device in server.all_physical_device_queue:
                if len(device_current_selected_client_ids[physical_device.physical_device_id]) == 0:
                    continue
                server.need_distribute_device += 1

        # ============= 下发========================
        for physical_device in server.all_physical_device_queue:
            if len(device_current_selected_client_ids[physical_device.physical_device_id]) == 0:
                continue
            # 发给client的信息
            server_2_client_data = {"model": server.trainer.get_model_dict(),
                                    "finished": False,
                                    "current_selected_client_ids":device_current_selected_client_ids[physical_device.physical_device_id],
                                    "global_epoch": global_epoch,
                                    **server.trainer.neet_to_send()}
            write_thread = WriteThread(server_2_client_data, physical_device) # 注意要深拷贝
            write_thread.start()

        write_finish.wait()
        write_finish.clear()        
        # while True:
        #     with server_lock:
        #         if server.current_state == "distributed":
        #             console.log("distributed to all clients",style="red")
        #             break # 全部发送完成
        #     time.sleep(1)
        # ========================================================
        
        # 准备接收uploaded信息
        while True:
            events = server.socket_manager.sel.select(timeout=0)  # 非阻塞调用,立即返回可用的文件描述符,而不等待
            for key, mask in events:
                if key.data is None:  # 服务端的listening socket；意味着有一个新的连接到来，需要注册
                    server.socket_manager.accept_wrapper(key.fileobj)
                else:
                    physical_device: physicalDevice = key.data  # 获得该物理设备对应的physicalDevice
                    server.socket_manager.sel.unregister(physical_device.sock) # 只需要读一次即可
                    if mask & selectors.EVENT_READ:  # 注册阶段只有读事件
                        read_thread = ReadThread(physical_device,False)
                        read_thread.start()

            with server_lock:
                if server.current_state == "uploaded":
                    console.log("received all the uploaded of current selected clients", style="red")
                    break  # 接收完成
        
        server.trainer.aggregate() # 聚合

        server.trainer.model = server.trainer.model.to(server.trainer.device) # 测试之前移动到gpu上
        accuracy,loss = evaluate(server.trainer.device, server.trainer.model, server.trainer.testloader)
        server.trainer.model = server.trainer.model.to("cpu") # 一定要转移到cpu上

        clientId_time_list = server.globel_epoch_training_time[global_epoch]
        client_time = []
        for clientId_time in clientId_time_list:
            assert clientId_time[0] in server.trainer.current_selected_client_ids
            client_time.append(clientId_time[1])
        server.global_time += max(client_time) # 全局时间加上最长设备时间
        console.log(f"the {global_epoch} global epoch acc is {accuracy}, used {max(client_time)}s. Current global time is {server.global_time}",style="bold red on white")
        if server.args['wandb']:
            server.experiment.log({"acc":accuracy},step = int(server.global_time))

        
    console.rule("finished training")
    server_2_client_data = {"finished": True,}
    for physical_device in server.all_physical_device_queue:
        write_thread = WriteThread(server_2_client_data, physical_device) # 注意要深拷贝
        write_thread.start()

    write_finish.wait()
    write_finish.clear()
    # while True:
    #     with server_lock:
    #         if server.current_state == "distributed":
    #             console.log("distributed to all clients")
    #             break # 全部发送完成
    #     time.sleep(1)



if __name__ == '__main__':
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    socket_manager = serversocket(args['server_ip'],args['server_port'])
    server = Server(args, socket_manager)
    console.log("initialized successfully")
    registerStage()
    trainingstage()
    server.close_all_sockets()

