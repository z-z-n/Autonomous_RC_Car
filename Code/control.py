# Author:   Zhining Zhang
# Task:     control the RC car;
# reference: https://docs.sunfounder.com/projects/picar-x/en/latest/python/python_move.html

import time
import socket
import re
import io
import struct
from picarx import Picarx
import picamera


def forward(msg):
    try:
        if len(msg) != 0 and re.match(r'\$[0-9]{2}#[0-9]{3}\$', msg):
            # order: $type#angle$, eg:$01#035$ turn left 35°
            msg = msg.strip('$')
            p = re.compile(r'#+')
            temp1 = p.split(msg)
            dir = int(temp1[0])
            angle = int(temp1[1])
            if dir == 0:  # forward
                print('go forward')
                px.forward(10)
            elif dir == 1:
                print('turn left: ', angle)
                px.set_dir_servo_angle(angle)
                px.forward(10)
            elif dir == 2:
                print('turn right: ', angle)
                px.set_dir_servo_angle(angle)
                px.forward(10)
            elif dir == 3:
                print('stop')
            elif dir == 4:
                print('others')
            time.sleep(0.5)
    finally:
        px.forward(0)


# create socket and bind host
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.0.210', 8000))
connection = client_socket.makefile('wb')
px = Picarx()

try:
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)  # pi camera resolution
        camera.framerate = 15  # 15 frames/sec
        time.sleep(2)  # give 2 secs for camera to initilize
        start = time.time()
        stream = io.BytesIO()

        # send jpeg format video stream
        for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            stream.seek(0)
            connection.write(stream.read())
            if time.time() - start > 600:
                break
            stream.seek(0)
            stream.truncate()
            '''
            # 移动
            msg = client_socket.recv(1024)
            msg = msg.decode()
            if msg != 0:
                print(msg)
                forward(msg)
            '''
    connection.write(struct.pack('<L', 0))
finally:
    connection.close()
    client_socket.close()