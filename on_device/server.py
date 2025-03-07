import json
import socket
import selectors
import random
import logging
import threading
import time
import queue
from rich.console import Console
from rich.padding import Padding
import wandb
import pickle
import multiprocessing
import yaml
import io
import struct
from pathlib import Path
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



console = Console()  # 终端输出对象
server_lock = threading.RLock()  # 多线程的stateInServer锁
print_lock = multiprocessing.RLock()  # 多进程的输出锁




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


class MyThread(threading.Thread):  # 每个线程与一个进程对应
    def __init__(self, event, multiprocessing_shared_queue):
        super().__init__()
        global server, server_lock
        self.server = server
        self.event = event
        self.server_lock = server_lock
        self.multiprocessing_shared_queue = multiprocessing_shared_queue

    def run(self):
        self.event.wait()  # 等待对应的子进程完成
        option, physical_device_id, value, client_2_server_data= self.multiprocessing_shared_queue.get()
        if option == "one register":  # 一个注册进程
            with self.server_lock:
                physical_device = server.all_physical_device_queue[physical_device_id]
                assert physical_device.physical_device_id == physical_device_id, f"the physical device id {physical_device_id} is not equal to physical_device.physical_device_id {physical_device.physical_device_id}"
                physical_device.name = value # 在子线程中进行修改，不能在子进程中修改
                server.wait_queue.put("register")
                if server.wait_queue.qsize() == server.need_connect_device:
                    server.current_state = 'registered'
                    server.wait_queue.queue.clear()

        elif option == "distributed":
            with self.server_lock:
                physical_device = server.all_physical_device_queue[physical_device_id]
                assert physical_device.physical_device_id == physical_device_id, f"the physical device id {physical_device_id} is not equal to physical_device.physical_device_id {physical_device.physical_device_id}"
                server.wait_queue.put("distributed")
                if server.wait_queue.qsize() == server.need_connect_device:
                    server.current_state = 'distributed' 
                    server.wait_queue.queue.clear()
        elif option == "finishDownload":  # 下发完成
            pass


class ReadProcess(multiprocessing.Process):
    def __init__(self, physical_device, print_lock, event, multiprocessing_shared_queue):
        super().__init__()
        self.physical_device = physical_device
        self.print_lock = print_lock
        self.event = event  # 事件
        self.multiprocessing_shared_queue = multiprocessing_shared_queue  # 与父进程的某个子线程进行信息传递的队列
        self._recv_buffer = b""  # 接收缓冲区
        self._jsonheader_len = None
        self.jsonheader = None
        self.request = None

    def _read(self):
        try:
            data = self.physical_device.sock.recv(20_971_520)  # read 20MB once
        except BlockingIOError:
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.")

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
        if not len(self._recv_buffer) >= content_len:
            return
        # 全部数据均已接收
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]
        self.request = decode(data)

        if self.request.get('action') == 'register':
            name = self.request.get('name')
            with self.print_lock:
                console.log(f"Received {name} register request", style="bold yellow")
            self.multiprocessing_shared_queue.put(("one register", self.physical_device.physical_device_id, name, self.request))  # 返回给server修改,第2个参数表示对应的message

        elif self.request.get('action') == 'upload':
            pass

    def run(self):
        with self.print_lock:
            console.log(
                f"start reading {self.physical_device.name}'s upload whose messageId is {self.physical_device.physical_device_id}" if self.physical_device.name != "" else "A new connection!")
        while True:
            self._read()
            if self._jsonheader_len is None:
                self.process_protoheader()
            if self._jsonheader_len is not None:
                if self.jsonheader is None:
                    self.process_jsonheader()
            if self.jsonheader:
                if self.request is None:
                    self.process_request()
                if self.request is not None:
                    break
        with self.print_lock:
            console.log(
                f"finish reading {self.physical_device.name}'s upload whose messageId is {self.physical_device.physical_device_id}" if self.physical_device.name != "" else "A new connection accepted!")
        self.event.set()  # 读进程完成，对应的父进程的子线程可以开始处理数据



class WriteProcess(multiprocessing.Process):
    def __init__(self, content_server_2_client, physical_device, print_lock, event, multiprocessing_shared_queue):
        super().__init__()
        self.content_server_2_client = content_server_2_client
        self.physical_device = physical_device
        self.print_lock = print_lock
        self.event = event
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
        response = encode(self.content_server_2_client)
        return response

    def run(self):
        with self.print_lock:
            console.log(f"start sending to {self.physical_device.name} whose messageId is {self.physical_device.physical_device_id}")
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
        self.multiprocessing_shared_queue.put(("distributed", self.physical_device.physical_device_id, self.physical_device.name, ""))
        with self.print_lock:
            console.log(f"Sent to {self.physical_device.name} whose messageId is {self.physical_device.physical_device_id}")
        self.event.set()




class Server:
    def __init__(self,need_connect_device, tot_client_nums, total_epoches, socket_manager):
        self.need_connect_device = need_connect_device
        self.current_client_nums = 0
        self.tot_client_nums = tot_client_nums
        self.current_epoches = 0
        self.total_epoches = total_epoches

        self.socket_manager = socket_manager

        self.all_physical_device_queue = [] # 记录所有物理设备的队列

        self.current_state = None # registered/distributed/download

        self.wait_queue = queue.Queue()  # 多线程共享队列， 存储线程的处理结果; 要注意清空
    
    def add_epoch(self):
        self.current_epoches += 1
    
    def add_client(self):
        self.current_client_nums += 1
    
    def finish(self):
        if self.current_epoches == self.total_epoches:
            return True
        else :
            return False
    



class physicalDevice:
    def __init__(self, sock, physical_device_id, addr):
        self.sock = sock
        self.physical_device_id = physical_device_id # socket的id，也对应着设备的id
        self.name = ""
        self.address = addr
    
    def __repr__(self):  # 日志输出用
        return f"\n{self.name}'s message id is {self.physical_device_id}"  # 为了适应日志输出，加上换行符


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

    
    def accept_wrapper(self, sock, physical_device_id):
        conn, addr = sock.accept()
        conn.setblocking(False)
        physical_device = physicalDevice(conn, physical_device_id, addr)
        self.sel.register(conn, selectors.EVENT_READ, data=physical_device)
        return physical_device

 
def registerStage(server):
    try:
        while True:
            events = server.socket_manager.sel.select(timeout=0)  # 非阻塞调用,立即返回可用的文件描述符,而不等待
            for key, mask in events:
                if key.data is None:  # 服务端的listening socket；意味着有一个新的连接到来，需要注册
                    physical_device:physicalDevice = server.socket_manager.accept_wrapper(key.fileobj, server.current_client_nums)
                    server.add_client()
                    server.all_physical_device_queue.append(physical_device)
                else:
                    physical_device: physicalDevice = key.data  # 获得该物理设备对应的physicalDevice
                    server.socket_manager.sel.modify(physical_device.sock, 0, data=physical_device)  # 将这个设备的sock在sel中的状态暂时挂起
                    if mask & selectors.EVENT_READ:  # 注册阶段只有读事件
                        event = multiprocessing.Event()
                        multiprocessing_shared_queue = multiprocessing.Queue()
                        read_process = ReadProcess(physical_device, print_lock, event, multiprocessing_shared_queue)
                        read_thread = MyThread( event,multiprocessing_shared_queue)
                        read_thread.start()
                        read_process.start()

            with server_lock:
                if server.current_state is not None:
                    console.log(f"all the clients has registered! The stateInServer.allClientMessageQueue is {server.all_physical_device_queue}")
                    break
        
        for physical_device in server.all_physical_device_queue:
            event = multiprocessing.Event()
            multiprocessing_shared_queue = multiprocessing.Queue()
            write_process = WriteProcess("received your files", physical_device, print_lock,event,multiprocessing_shared_queue)
            write_thread = MyThread(event, multiprocessing_shared_queue)
            write_thread.start()
            write_process.start()
        
        while True:
            with server_lock:
                if server.current_state == "distributed":
                    console.log("distributed to all clients")
                    break
            
            

    except Exception:
        print("Something wrong in register stage")



if __name__ == '__main__':
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    socket_manager = serversocket(args['server_ip'],args['server_port'])
    server = Server(args['need_connect_device'], args['client_num'], args['global_epoch'], socket_manager)
    registerStage(server)
    for physical_device in server.all_physical_device_queue:
        server.socket_manager.sel.unregister(physical_device.sock)
        physical_device.sock.close()
    server.socket_manager.sel.close()

