import socket
import threading


class SocketWriter:
    def __init__(self, destinationPort=50001):
        self.HOST = "127.0.0.1"
        self.PORT = destinationPort

        self.connect()

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.HOST, self.PORT))

    def flush(self):
        self.sock.close()
        self.connect()

    def write(self, data):
        self.sock.sendall(bytes(data, "utf-8"))

    def __del__(self):
        self.sock.close()


class SocketListener:

    def __init__(self, destinationPort=50001):
        self.HOST = "127.0.0.1"
        self.PORT = destinationPort

        self.connect()

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.HOST, self.PORT))

    def flush(self):
        self.sock.close()
        self.connect()

    def write(self, data):
        self.sock.sendall(bytes(data, "utf-8"))

    def __del__(self):
        self.sock.close()
