

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Lock(metaclass=Singleton):
    def __init__(self, initallyLocked=False):
        print("Lock created")
        self._locked = initallyLocked

    def lock(self):
        self._locked = True
        print("locked")

    def unlock(self):
        self._locked = False
        print("unlocked")

    def isLocked(self):
        return self._locked

    def toggle(self):
        print("toggled lock")
        self._locked = not self._locked

