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
from torch.func import grad, vmap
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
console = Console()
client_lock = threading.RLock()
print_lock = multiprocessing.RLock()
class BaseClient:
    def __init__(self, client_id, train_index, batch_size):
        self.client_id = client_id
        self.train_set_index = np.array(train_index)
        self.train_set_len = len(train_index)
        self.participation_times = 0
        self.batch_size = batch_size
        self.training_time = 0
        self.pretrained_accuracy = 0
        self.accuracy = 0
        self.loss = 0.0
        self.grad = None
        self.buffer = None
        self.training_time_record = {}
        self.train_set_label = None
        self.label_num_distribution = {}
        self.buffer = {}
        self.buffer_idx = []
        self.buffer_size = 100
        self.new_num_train_samples = 0.0
        self.label_weight = {}
        self.weights_tensor = None
        self.new_weight4aggregation = 0.0
    def participate_once(self):
        self.participation_times += 1
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
        self.model = MODEL_DICT[self.args["model"]](self.data_num_classes)
        self.current_client_instance = None
        self.trainset = DATASETS[self.args['dataset']](PROJECT_DIR / "data" / self.args["dataset"], "train")
        self.train_sampler = CustomSampler(list(range(len(self.trainset))))
        self.trainloader = DataLoader(Subset(self.trainset, list(range(len(self.trainset)))), self.args["batch_size"], num_workers=2,collate_fn = DATASETS_COLLATE_FN[self.args['dataset']], persistent_workers=True,
                                      sampler=self.train_sampler,)
        self.local_epoch = self.args["local_epoch"]
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none').to(self.device)
        self.optimizer = None
        self.timer = Timer()
    def load_dataset(self):
        self.trainloader.sampler.set_index(self.current_client_instance.train_set_index)
        self.trainloader.batch_sampler.batch_size = self.current_client_instance.batch_size
    def set_parameters(self,model_parameters):
        self.model.load_state_dict(model_parameters)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),lr=self.args["lr"],momentum=self.args["momentum"],weight_decay=self.args["weight_decay"],)
    def start(self,global_epoch, client_instance, model_parameters):
        self.timer.start()
        self.current_client_instance = client_instance
        self.set_parameters(model_parameters)
        self.load_dataset()
        self.local_train()
        self.model = self.model.to("cpu")
        current_client_instance_model_dict = {key: copy.deepcopy(value) for key, value in self.model.state_dict().items()}
        self.timer.stop()
        self.current_client_instance.training_time_record[global_epoch] = self.timer.times[-1]
        self.current_client_instance.participate_once()
        return current_client_instance_model_dict, self.current_client_instance.training_time_record[global_epoch]
    def get_client_grad(self , client_instance , factor):
        self.current_client_instance = client_instance
        self.model.load_state_dict(self.current_client_instance.model_dict)
        self.trainloader.sampler.set_index(self.current_client_instance.train_set_index)
        self.trainloader.batch_sampler.batch_size = self.current_client_instance.batch_size if self.args['model'] != "vit" else int(self.current_client_instance.batch_size // 2)
        self.model.train()
        __NormBase = torch.nn.BatchNorm2d.__mro__[2]
        for module in self.model.modules():
            if isinstance(module,__NormBase):
                module.track_running_stats = False
        if self.args['model'] != "vit":
            trainable_parameters = [p for p in self.model.parameters()]
        else:
            trainable_parameters = []
            for name, param in self.model.named_parameters():
                if 'head' in name or 'lora' in name or 'Prompt' in name:
                    trainable_parameters.append(param)
        cnt = 0
        gradient = []
        self.optimizer.zero_grad()
        for inputs, targets in self.trainloader:
            if isinstance(inputs,torch.Tensor):
                inputs = inputs.to(self.device, non_blocking=True)
            else:
                inputs = [tensor.to(self.device, non_blocking=True) for tensor in inputs]
            targets = targets.to(self.device,non_blocking=True)
            outputs = self.model(inputs)
            loss = self.criterion_(outputs, targets).sum()
            if cnt == 0:
                gradient = torch.autograd.grad(loss,trainable_parameters)
            else:
                gradient =[torch.sum(torch.stack(grad,dim=-1),dim=-1) for grad in zip(gradient, torch.autograd.grad(loss,trainable_parameters))]
            cnt += 1
        __NormBase = torch.nn.BatchNorm2d.__mro__[2]
        for module in self.model.modules():
            if isinstance(module,__NormBase):
                module.track_running_stats = True
        return [factor * _grad for _grad in gradient]
    def update_client_buffer(self,client_instance,global_gradient,batch_size):
        self.current_client_instance = client_instance
        self.model.load_state_dict(self.current_client_instance.model_dict)
        self.model.train()
        self.trainloader.sampler.set_index(self.current_client_instance.train_set_index)
        self.trainloader.batch_sampler.batch_size = batch_size
        __NormBase = torch.nn.BatchNorm2d.__mro__[2]
        for module in self.model.modules():
            if isinstance(module,__NormBase):
                module.track_running_stats = False
        params = dict(self.model.named_parameters())
        if self.args['model'] != "vit":
            trainable_parameters = [p for p in self.model.parameters()]
        else:
            trainable_parameters = []
            for name, param in self.model.named_parameters():
                if 'head' in name or 'lora' in name or 'Prompt' in name:
                    trainable_parameters.append(param)
        for inputs,targets in self.trainloader:
            result=torch.zeros(len(targets),dtype=torch.float32,device=self.device)
            if isinstance(inputs,torch.Tensor) and self.args['model'] != "vit":
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device,non_blocking=True)
                gradients: dict[str:torch.tensor] = vmap(grad(self.loss_fn), in_dims=(None, 0, 0),randomness='same')(params, inputs, targets)
                for local_grad,global_grad in zip(gradients.values(),global_gradient):
                    result += torch.sum(local_grad * global_grad,dim=tuple(range(1,local_grad.ndim)))
            else:
                self.optimizer.zero_grad()
                if isinstance(inputs,torch.Tensor):
                    inputs = inputs.to(self.device, non_blocking=True)
                else:
                    inputs = [tensor.to(self.device, non_blocking=True) for tensor in inputs]
                targets = targets.to(self.device,non_blocking=True)
                outputs = self.model(inputs)
                loss = self.criterion_(outputs, targets).mean()
                _grad = torch.autograd.grad(loss,trainable_parameters)
                self.optimizer.zero_grad()
                for local_grad,global_grad in zip(_grad,global_gradient):
                    result += torch.sum(local_grad * global_grad)
            self.trainloader.dataset.update(result)
        del trainable_parameters
        value:np.array = self.trainloader.dataset.get_value(self.current_client_instance.train_set_index)
        self.current_client_instance.buffer_idx = []
        for label in self.current_client_instance.label_weight.keys():
            indices=np.where(self.current_client_instance.train_set_label == label)[0]
            value_at_indices = value[indices]
            sorted_indices=indices[np.argsort(value_at_indices)]
            self.current_client_instance.buffer_idx.extend(self.current_client_instance.train_set_index[sorted_indices[-self.current_client_instance.buffer_size[label]:]].tolist())
        for module in self.model.modules():
            if isinstance(module,__NormBase):
                module.track_running_stats = True
        self.current_client_instance.train_set_len = len(self.current_client_instance.buffer_idx)
    def loss_fn(self,params, data, target):
        output = torch.func.functional_call(self.model, params, (data.unsqueeze(0),))
        loss = self.criterion(output, target.unsqueeze(0))
        return loss
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
                loss = self.criterion(outputs, targets).mean()
                loss.backward()
                self.optimizer.step()
        torch.cuda.synchronize()
    def local_train(self):
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
            data = self.client.socket_manager.sock.recv(20_971_520)
        except BlockingIOError:
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.")
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
        if not len(self._recv_buffer) == content_len:
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
            if not self.finishedRead:
                time.sleep(1.0)
class MyThread(threading.Thread):
    def __init__(self,):
        super().__init__()
        global client , client_lock, print_lock
        self.client = client
        self.client_lock = client_lock
        self.print_lock = print_lock
        self.daemon = True
    def run(self):
        while True:
            need_to_send = self.client.need_to_send_queue.get()
            write_process = WriteProcess(self.client.server_ip_port, need_to_send , self.print_lock)
            write_process.start()
            write_process.join()
            with self.client_lock:
                self.client.need_to_send_num -= 1
class WriteProcess(multiprocessing.Process):
    def __init__(self, server_ip_port, need_to_send, print_lock):
        super().__init__()
        self.need_to_send = need_to_send
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)
        self.sock.connect_ex(server_ip_port)
        self.print_lock = print_lock
        self._send_buffer = b""
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
        message = self._create_message(response)
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
        self.sock.connect_ex((self.server_ip,self.server_port))
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
        message = self._create_message(response)
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
            self.sock = None
class Client:
    def __init__(self, args, socket_manager, name):
        self.args = args
        self.socket_manager = socket_manager
        self.server_ip_port = (self.socket_manager.server_ip , self.socket_manager.server_port)
        self.name = name
        self.received_data = None
        self.need_to_send_queue = queue.Queue()
        self.need_to_send_num = 0
        self.client_ids = []
        self.clientId_dataIndex = {}
        self.client_instances_dict = {}
        self.trainer = Trainer(self.args)
        self.current_epoch_transmission = 0
        self.current_selected_client_ids = []
        self.model = None
def read_from_server():
    read_thread = ReadThread()
    read_thread.start()
    read_thread.join()
def run():
    global client, client_lock
    client.socket_manager.send()
    read_from_server()
    client.client_ids = client.received_data["clients"]
    client.clientId_dataIndex = client.received_data["data_indices"]
    console.log(f"device {client.name} need to train {client.client_ids}" , style='red')
    for client_id in client.client_ids:
        client.client_instances_dict[client_id] = BaseClient(client_id, client.clientId_dataIndex[client_id], client.args["batch_size"])
    console.log("clients has been initialized successfully")
    client_2_server_data = dict(name=client.name,action= "check") 
    with client_lock:
        client.need_to_send_num += 1
    client.need_to_send_queue.put(client_2_server_data)
    while True:
        with client_lock:
            if client.need_to_send_num == 0:
                break
        time.sleep(1)
    while True:
        read_from_server() 
        if client.received_data['finished']:
            break
        console.rule(f"start {client.received_data['global_epoch']}",style='red')
        console.log(f"transmission time is {client.received_data['server_2_client_time']}")
        client.model = client.received_data['model']
        client.current_epoch_transmission = client.received_data['server_2_client_time']
        client.current_selected_client_ids = client.received_data['current_selected_client_ids']
        console.log(f"need to train {client.current_selected_client_ids} in global epoch {client.received_data['global_epoch']}")
        for client_id in client.current_selected_client_ids:
            assert client_id in client.client_ids , f"{client_id} do not belongs to the device" 
            current_client_instance_model_dict, training_time = client.trainer.start(client.received_data['global_epoch'],client.client_instances_dict[client_id],client.model)
            client_2_server_data = dict(
                                        name=client.name,
                                        action="upload",
                                        client_id=client_id,
                                        client_model = current_client_instance_model_dict,
                                        weight = client.client_instances_dict[client_id].train_set_len,
                                        s2c_training_time = training_time + client.current_epoch_transmission,
                                        **client.client_instances_dict[client_id].neet_to_send())
            console.log(f"{client_id} has finished training, using {training_time}s")
            with client_lock:
                client.need_to_send_num += 1
            client.need_to_send_queue.put(client_2_server_data)
        console.log(f"{client.name} has finished local training of all selected clients")
        while True:
            with client_lock:
                if client.need_to_send_num == 0:
                    break
            time.sleep(1)
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
