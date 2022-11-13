from cmd import Cmd
from utils.window import Window
from console.exceptions import StopConsole


class CameraConsole(Cmd):
    def __init__(self, window: Window, prompt: str):
        super().__init__()
        self.window = window
        self.prompt = prompt

    # Command EXIT

    def do_exit(self, _):
        raise StopConsole

    def help_exit(self):
        print('Volta para o menu principal')
