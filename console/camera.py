import numpy as np
from cmd import Cmd
from utils.camera import Camera
from utils.window import Window
from console.exceptions import StopConsole


class CameraConsole(Cmd):
    def __init__(self, window: Window, prompt: str):
        super().__init__()
        self.window = window
        self.prompt = prompt
        self.__print_camera()

    def __print_camera(self):
        camera = self.window.scene.updateCamera or self.window.scene.camera
        print(f'eye: {camera.position}')
        print(f'at: {camera.at}')
        print(f'rotation: {camera.rotation}º')
        print(f'window size: {camera.windowSize}')
        print(f'distance: {camera.distance}')
        if camera.perpendicular: print('Projeção ortográfica')
        else: print('Projeção perspectiva')

    def do_show(self, _):
        self.__print_camera()

    def do_eye(self, args):
        args = args.split()
        try:
            eye = np.array(list(map(float, args[:3])), dtype=np.float64)
            assert len(eye) == 3
        except Exception as ex:
            print(ex)
            print("Digite coordenadas válidas!")
            return

        old_camera = self.window.scene.updateCamera or self.window.scene.camera
        self.window.scene.pushCamera(Camera(
            old_camera.resolution,
            eye,
            old_camera.at,
            rotation=old_camera.rotation,
            perpendicular=old_camera.perpendicular,
            n_threads=old_camera.n_threads,
            distance=old_camera.distance,
            windowSize=old_camera.windowSize
        ))
        self.__print_camera()

    def do_at(self, args):
        args = args.split()
        try:
            at = np.array(list(map(float, args[:3])), dtype=np.float64)
            assert len(at) == 3
        except Exception as ex:
            print(ex)
            print("Digite coordenadas válidas!")
            return

        old_camera = self.window.scene.updateCamera or self.window.scene.camera
        self.window.scene.pushCamera(Camera(
            old_camera.resolution,
            old_camera.position,
            at,
            rotation=old_camera.rotation,
            perpendicular=old_camera.perpendicular,
            n_threads=old_camera.n_threads,
            distance=old_camera.distance,
            windowSize=old_camera.windowSize
        ))
        self.__print_camera()

    def do_rotation(self, args):
        args = args.split()
        try:
            rotation = int(args[0]) % 360
        except Exception as ex:
            print(ex)
            print("Digite uma rotação válida!")
            return

        old_camera = self.window.scene.updateCamera or self.window.scene.camera
        self.window.scene.pushCamera(Camera(
            old_camera.resolution,
            old_camera.position,
            old_camera.at,
            rotation=rotation,
            perpendicular=old_camera.perpendicular,
            n_threads=old_camera.n_threads,
            distance=old_camera.distance,
            windowSize=old_camera.windowSize
        ))
        self.__print_camera()

    def do_window(self, args):
        args = args.split()
        try:
            windowSize = np.array(list(map(int, args[:2])), dtype=np.int32)
            assert len(windowSize) == 2
        except Exception as ex:
            print(ex)
            print("Digite um tamanho de janela válido!")
            return

        old_camera = self.window.scene.updateCamera or self.window.scene.camera
        self.window.scene.pushCamera(Camera(
            old_camera.resolution,
            old_camera.position,
            old_camera.at,
            rotation=old_camera.rotation,
            perpendicular=old_camera.perpendicular,
            n_threads=old_camera.n_threads,
            distance=old_camera.distance,
            windowSize=windowSize
        ))
        self.__print_camera()

    def do_distance(self, args):
        args = args.split()
        try:
            distance = float(args[0])
        except Exception as ex:
            print(ex)
            print("Digite uma distância válida!")
            return

        old_camera = self.window.scene.updateCamera or self.window.scene.camera
        self.window.scene.pushCamera(Camera(
            old_camera.resolution,
            old_camera.position,
            old_camera.at,
            rotation=old_camera.rotation,
            perpendicular=old_camera.perpendicular,
            n_threads=old_camera.n_threads,
            distance=distance,
            windowSize=old_camera.windowSize
        ))
        self.__print_camera()

    def do_perspective(self, _):
        old_camera = self.window.scene.updateCamera or self.window.scene.camera
        self.window.scene.pushCamera(Camera(
            old_camera.resolution,
            old_camera.position,
            old_camera.at,
            rotation=old_camera.rotation,
            perpendicular=False,
            n_threads=old_camera.n_threads,
            distance=old_camera.distance,
            windowSize=old_camera.windowSize
        ))
        self.__print_camera()

    def do_ortografica(self, _):
        old_camera = self.window.scene.updateCamera or self.window.scene.camera
        self.window.scene.pushCamera(Camera(
            old_camera.resolution,
            old_camera.position,
            old_camera.at,
            rotation=old_camera.rotation,
            perpendicular=True,
            n_threads=old_camera.n_threads,
            distance=old_camera.distance,
            windowSize=old_camera.windowSize
        ))
        self.__print_camera()

    # Command EXIT

    def do_exit(self, _):
        raise StopConsole

    def help_exit(self):
        print('Volta para o menu principal')
