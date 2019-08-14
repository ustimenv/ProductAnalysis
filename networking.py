import socket


class SocketWriter:

    def __init__(self, destinationPort=50001):
        self.HOST = "127.0.0.1"
        self.PORT = destinationPort

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((self.HOST, self.PORT))

    def send(self, data):
        try:
            self.sock.sendall(bytes(data, "utf-8"))
            return 0
        except:
            return data

    def __del__(self):
        self.sock.close()
