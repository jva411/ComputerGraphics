import time
from cmd import Cmd
from utils.window import Window
from console.camera import CameraConsole
from console.lights import LightsConsole
from console.exceptions import StopConsole


class Console(Cmd):
    LIST_OPTIONS = ['objects', 'lights']
    SELECT_OPTIONS = ['lights', 'camera']

    def __init__(self, window: Window, prompt: str):
        super().__init__()
        self.window = window
        self.prompt = prompt
        self.__waitRender()

    def __waitRender(self):
        print()
        animationTick = 0
        dots = 0
        while self.window.scene.loaded is False:
            animationTick += 1
            if animationTick == 6:
                print('\rRenderizando cenário' + '.'*(dots+1) + ' '*(2-dots), end='')
                dots = (dots+1) % 3
                animationTick = 0
            time.sleep(0.1)
        print('\rCenário renderizado!   ')

    # Command RENDER

    def do_render(self, arg):
        self.window.rerender()
        self.__waitRender()


    def help_render(self):
        print('Renderiza o cenário novamente')

    # Command LIST

    def __printObjects(self):
        print('Ojbetos:')
        for idx, object in enumerate(self.window.scene.objects):
            print(idx+1, '-', object.getDescription().split('\n')[0])

    def __printLights(self):
        print('Luzes:')
        for idx, light in enumerate(self.window.scene.lights):
            print(idx+1, '-', light.__class__.__name__)


    def do_list(self, arg):
        match arg:
            case 'objects': return
            case 'lights': return
            case _:
                self.__printObjects()
                self.__printLights()
                return

    def help_list(self):
        print('Lista todos todos os objetos e luzes do cenário')
        print('list objects - Lista todos os objetos do cenário')
        print('list lights - Lista todos os objetos do cenário')

    # Command SELECT

    def do_select(self, arg: str):
        args = arg.lower().split(' ')
        type = args[0]

        try:
            match type:
                case 'camera':
                    CameraConsole(self.window, self.prompt + 'camera>').cmdloop()
                case 'lights':
                    LightsConsole(self.window, self.prompt + 'light>').cmdloop()
                case _:
                    return
        except StopConsole:
            return

    def complete_select(self, text, *_):
        if text:
            return [option for option in self.SELECT_OPTIONS if option.startswith(text)]

        return self.SELECT_OPTIONS

    def help_select(self):
        print('use select <objet/light> <id> para manipular um objeto do cenário')
        print('use select <camera> para manipular a câmera do cenáio')

    # Command EXIT

    def do_exit(self, _):
        self.window.close()
        exit(0)

    def help_exit(self):
        print('Encerra o programa')
