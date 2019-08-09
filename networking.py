import socket


class SocketWriter:

    def __init__(self, destinationPort=50001):
        self.SOCKET_IP = "localhost"
        self.SOCKET_PORT = destinationPort

    def send(self, data):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("localhost", self.SOCKET_PORT))
        self.sock.sendall(bytes(data, "utf-8"))
        self.sock.close()

