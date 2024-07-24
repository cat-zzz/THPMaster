"""
@File   : human_gui.py
@Desc   : 用户交互界面
@Author : gql
@Date   : 2023/5/29 14:27
"""
import socket
import threading
import tkinter as tk
import argparse


def connect_server(name, ip='127.0.0.1', port=10001):
    """
    建立连接并发送名称

    :return: network
    """
    # name = "big_cat"  # 名称
    # ip_port = ("127.0.0.1", 10001)
    ip_port = (ip, port)
    print("---开始连接---")
    s = socket.socket()
    s.connect(ip_port)
    print("---连接成功---")
    reply = s.recv(1024).decode("ASCII")
    if reply == "name":
        print("server:", reply)
        s.send(name.encode())
    return s


class GUI:
    def __init__(self, name, ip, port):
        self.name = name
        self.root = tk.Tk()
        self.count = 1  # 记录服务端发来的指令数
        self.episode = 1  # 记录对局数
        self.root.title("客户端--by gql")
        self.root.geometry("800x500+430+120")
        self.client_name = None
        self.input = None
        self.server_list_msg = None
        self.client_input = None
        self.client_send_btn = None
        self.fold_btn = None
        self.interface()
        self.socket = connect_server(name, ip, port)
        self.display_msg('name', name)
        # 开启一个线程，专门用于接受服务端消息
        t1 = threading.Thread(name='t1', target=self.receive_server_msg, daemon=True)  # 子线程
        t1.start()  # 启动子线程

    def interface(self):
        """
        用户界面
        """
        self.client_name = tk.Label(self.root, text=self.name)
        self.client_name.pack()
        self.server_list_msg = tk.Listbox(self.root, width=52, height=20)
        self.server_list_msg.pack()
        self.input = tk.StringVar()
        self.client_input = tk.Entry(self.root, width=52, textvariable=self.input)
        self.client_input.bind('<Return>', self.send_cmd_event)
        self.client_input.pack()
        self.client_send_btn = tk.Button(self.root, text="发送指令", command=self.send_cmd_event)
        self.client_send_btn.pack()
        self.fold_btn = tk.Button(self.root, text="fold指令", command=self.send_fold_cmd_event)
        self.fold_btn.pack()

    def send_cmd_event(self, event=''):
        """
        发送客户端指令，绑定Button事件和输入框的回车事件
        @:param event: Button事件不需要参数，但绑定回车事件需要参数
        """
        client_cmd = self.input.get()
        if client_cmd == '':
            return
        self.socket.sendall(client_cmd.encode("ASCII"))
        self.display_msg('client', client_cmd)
        self.input.set('')

    def send_fold_cmd_event(self, event=''):
        self.socket.sendall("fold".encode("ASCII"))
        self.display_msg('client', "fold")
        self.input.set('')

    def start(self):
        self.root.mainloop()

    def display_msg(self, sender, msg):
        if sender == '':
            s = str(self.count) + '  ' + str(msg)
        else:
            s = str(self.count) + '  ' + str(sender) + ': ' + str(msg)
        print(s)
        self.server_list_msg.insert(tk.END, s)
        self.server_list_msg.see(tk.END)  # 让列表框下拉到最后
        self.count += 1

    def receive_server_msg(self):
        """
        用于接受服务端消息的子线程
        """
        s = self.socket
        while True:
            reply = s.recv(1024).decode()
            self.display_msg('server', reply)
            if reply[0] == 'e':
                self.display_msg('', '第{}局结束'.format(self.episode))
                self.episode += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='Thinking in THP')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=10001)
    args = parser.parse_args()
    # args_dict = args.__dict__
    print(args.name, args.ip, args.port)
    client = GUI(args.name, args.ip, args.port)
    client.start()
