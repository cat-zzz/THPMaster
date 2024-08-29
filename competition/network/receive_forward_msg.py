"""
@File   : receive_forward_msg.py
@Desc   : 接受并转发服务端消息
@Author : gql
@Date   : 2023/5/30 14:10
"""
import queue
import socket
import threading
import time


class ReceiveMsgThread(threading.Thread):
    def __init__(self, s: socket, msg_queue: queue.Queue):
        super().__init__()
        self.s = s
        self.msg_queue = msg_queue
        self.thread_stop = False

    def run(self) -> None:
        while not self.thread_stop:
            try:
                reply = self.s.recv(1024).decode()
            except Exception as e:
                print("终止接收消息线程")
                print("退出程序")
                exit(0)
            # reply = str(i) + '--thread'
            print("接收消息线程-->来自服务端的原始字符串:", reply)
            self.msg_queue.put(reply, block=True)

    def stop(self, thread_stop: bool):
        self.thread_stop = thread_stop


def connect_server(name, ip_port=("127.0.0.1", 10001)):
    """
    与服务端建立连接
    """
    print("---开始连接---")
    s = socket.socket()
    s.settimeout(120)

    s.connect(ip_port)
    print("---连接成功---")
    reply = s.recv(1024).decode("ASCII")
    if reply == "name":
        print("server:", reply, name)
        s.send(name.encode())
    return s


if __name__ == '__main__':
    q = queue.Queue(0)
    msg_thread = ReceiveMsgThread(None, q)
    msg_thread.start()
    r = q.get(block=True)
    print("r:", r)
    time.sleep(8)
    r = q.get(block=True)
    print("r:", r)
    r = q.get(block=True)
    print("r:", r)
    r = q.get(block=True)
    print("r:", r)
