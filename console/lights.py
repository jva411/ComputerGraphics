import numpy as np
from cmd import Cmd
from utils.camera import Camera
from utils.window import Window
from console.exceptions import StopConsole


class LightsConsole(Cmd):
    def __init__(self, window: Window, prompt: str):
        super().__init__()
        self.window = window
        self.prompt = prompt
        self.__print_lights()

    def __print_lights(self):
        for idx, light in enumerate(self.window.scene.lights):
            print(f"{idx} - {'ligada   ' if light.on else 'desligada'} - {light.__class__.__name__}")

    def do_on(self, args):
        args = args.split()
        try:
            idx = int(args[0])
            light = self.window.scene.lights[idx]
        except:
            print("Digite uma luz válida!")
            return

        light.on = True
        self.__print_lights()

    def do_off(self, args):
        args = args.split()
        try:
            idx = int(args[0])
            light = self.window.scene.lights[idx]
        except:
            print("Digite uma luz válida!")
            return

        light.on = False
        self.__print_lights()

    # Command EXIT

    def do_exit(self, _):
        raise StopConsole

    def help_exit(self):
        print('Volta para o menu principal')
