import json
import sys
import socket
import traceback
import pickle
import argparse
import yaml
# 输出设置----------------------------------------------------------
from rich.console import Console
from rich.padding import Padding
from pathlib import Path
import sys
import io
import struct
import threading
import multiprocessing
import queue


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
client_lock = threading.RLock()  # 多线程的client访问锁
print_lock = multiprocessing.RLock()  # 多进程的输出锁



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
        super().__init__(self,)
        global client 
        self.client = client
        self._recv_buffer = b""
        self._jsonheader_len = None
        self.jsonheader = None
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
        if not len(self._recv_buffer) >= content_len:  # 还没收到完整数据
            return
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]
        self.client.received_data = decode(data)  # 反序列化
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
    def __init__(self, event, multiprocessing_shared_queue):
        super().__init__()
        global client
        self.client = client
        self.event = event
        self.multiprocessing_shared_queue = multiprocessing_shared_queue

    def run(self):
        self.event.wait()  # 等待对应的子进程完成
        option, = self.multiprocessing_shared_queue.get()
        if option == "uploaded":  # 一个注册进程
            with client_lock:
                self.client.socket_in_use = True
            


class WriteProcess(multiprocessing.Process):
    def __init__(self, socket, need_to_send, print_lock, event, multiprocessing_shared_queue):
        super().__init__()
        self.need_to_send = need_to_send
        self.sock = socket
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
        response = encode(self.need_to_send)
        return response

    def run(self):
        with self.print_lock:
            console.info(f"start sending to server")
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
        self.multiprocessing_shared_queue.put(("uploaded"))
        with self.print_lock:
            console.info(f"send to server successfully")
        self.event.set()

    


class clientsocket:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)
        self.sock.connect_ex((self.server_ip,self.server_port)) # 创建一个连接

def create_content(name, action, value):  # 这个就是上传的内容格式

    return dict(name=name,
                action=action)

class Client:
    def __init__(self, socket_manager, name):
        self.socket_manager = socket_manager
        self.name = name 

        self.communication_buffer = None

        self.socket_in_use = False
        self.need_to_send_queue = queue.Queue()

def send_2_server(client):
    with client_lock:
        client.socket_in_use = True
        content_client_2_server = client.need_to_send_queue.get()
    event = multiprocessing.Event()
    multiprocessing_shared_queue = multiprocessing.Queue()
    write_thread = MyThread(client, event, multiprocessing_shared_queue)
    write_process = WriteProcess(client.socket_manager.sock, content_client_2_server,print_lock,event,multiprocessing_shared_queue)
    write_thread.start()
    write_process.start()
    read_thread = ReadThread()
    read_thread.start()
    read_thread.join()




def register(client):
    content_client_2_server = create_content(client.name,"register")
    client.need_to_send_queue.put(content_client_2_server)
    send_2_server(client)


if __name__ == '__main__':
    parser = get_argparser().parse_args()
    with open(parser.config_path, 'r') as file:
        args = yaml.safe_load(file)
    if args["set_seed"]:
        fix_random_seed(args["seed"])
    
    socket_manager = clientsocket(args['server_ip'], args['server_port'])
    client = Client(socket_manager, parser.name)
    register(client)
    client.socket_manager.sock.close()  # 关闭socket
