import time
from cmd import Cmd
from utils.window import Window


commands = {
    'list': {
        'help': 'lista os objetos e as luzes do cenário',
        'commands' : {
            'objects': {
                'help': 'lista os objetos do cenário',
                'execute': None
            }
        },
        'execute': None
    }
}


class Console(Cmd):
    LIST_OPTIONS = ['objects', 'lights']

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
        self.window.scene.loaded = False
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

    # Command QUIT

    def do_quit(self, _):
        self.window.close()
        exit(0)
