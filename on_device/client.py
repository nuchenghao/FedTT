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
import queue
import selectors
import time

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
        console.log(f"transmission time is {server_2_client_time}")
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

def create_content(name, action):  # 这个就是上传的内容格式

    return dict(name=name,
                action=action)

class Client:
    def __init__(self, socket_manager, name):
        self.socket_manager = socket_manager
        self.server_ip_port = (self.socket_manager.server_ip , self.socket_manager.server_port)
        self.name = name 
        self.received_data = None

        self.need_to_send_queue = queue.Queue()
        self.need_to_send_num = 0


def read_from_server():
    read_thread = ReadThread()
    read_thread.start()
    read_thread.join()



def run():
    global client, client_lock
    # ============== register================
    client.socket_manager.send()
    read_from_server()
    print(client.received_data)

    client_2_server_data = create_content(client.name, "check")
    with client_lock:
        client.need_to_send_num += 1
    client.need_to_send_queue.put(client_2_server_data)
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
    client = Client(socket_manager, parser.name)
    write_daemon_thread = MyThread()
    write_daemon_thread.start()
    
    run()

    client.socket_manager.close()
