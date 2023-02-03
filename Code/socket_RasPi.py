# Author:   Zhining Zhang
# Task:     Raspberry-Pi socket communication; Pi as server;

import socket
import time

print("Server is on!")
# socket.AF_INET用于服务器与服务器之间的网络通信
# socket.SOCK_STREAM代表基于TCP的流式socket通信
mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = "192.168.1.104" # socket.gethostname()
server_port = 8888
# 绑定端口，告诉别人，这个端口我使用了，其他人别用了
mySocket.bind((server_ip, server_port))
# 监听这个端口，可连接最多10个设备
mySocket.listen(10)

while True:
    print("Waiting connection!")
    client, address = mySocket.accept()
    print("New connection!")
    print("IP is %s", address[0])
    print("port is %d\n", address[1])
    while True:
        msg = 0
        try:
            client.settimeout(10)  # 设置10s时限
            msg = client.recv(1024)
            msg = msg.decode()
            if msg != 0:
                print(msg)
        except socket.timeout:  # 超时
            print('time out!')
            client.close()  # 关闭连接
            break
    break
