# Author:   Zhining Zhang
# Task:     PC-side socket communication; PC as client;

import socket


class connect_Raspberry:
    def __init__(self, host, port):
        print("Client is on")
        # 套接字接口socket interface
        self.mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置ip和端口
        self.hostIp = host
        self.port = port

        try:
            # connecting to server
            self.mySocket.connect((self.hostIp, self.port))
            print("Connected to server!")
        except:
            # failed
            print('Connection failed!')

    def send(self, words):
        # send message
        msg = words
        # 编码发送
        self.mySocket.send(msg)
        print("sent successfully!")

    def close(self):
        self.mySocket.close()
        print("Lost connection!\n")
        exit()


# demo
# 设置ip和端口(IP为树莓派的IP地址)
# Set ip and port (IP is the IP address of Raspberry Pi)
myRaspConnection = connect_Raspberry('192.168.1.104', 8888)
myRaspConnection.send("hello world!")
