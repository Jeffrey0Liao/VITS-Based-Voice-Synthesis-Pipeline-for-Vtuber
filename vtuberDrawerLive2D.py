import time
import math

import yaml
import numpy as np

import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from psd_tools import PSDImage
from rimo_utils import matrix


def extractLayers(psd):
    layerList = []
    def dfs(layer, path=''):
        if layer.is_group():
            for i in layer:
                dfs(i, path + layer.name + '/')
        else:
            a, b, c, d = layer.bbox
            npdata = layer.numpy()
            npdata[:, :, 0], npdata[:, :, 2] = npdata[:, :, 2].copy(), npdata[:, :, 0].copy()
            layerList.append({'name': path + layer.name, 'position': (b, a, d, c), 'npdata': npdata})
    for layer in psd:
        dfs(layer)
    return layerList, psd.size


def openglDrawLoop(layerList, psd_size):
    def generateTexture(img):
        w, h = img.shape[:2]
        d = 2**int(max(math.log2(w), math.log2(h)) + 1)
        texture = np.zeros([d, d, 4], dtype=img.dtype)
        texture[:w, :h] = img
        return texture, (w / d, h / d)

    vtuber_size = 512, 512

    glfw.init()
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(*vtuber_size, 'Vtuber', None, None)
    glfw.make_context_current(window)
    glViewport(0, 0, *vtuber_size)

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    for layer in layerList:
        textureIndex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, textureIndex)
        texture, textureCoordinate = generateTexture(layer['npdata'])
        width, height = texture.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_FLOAT, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
        layer['textureIndex'] = textureIndex
        layer['textureCoordinate'] = textureCoordinate

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        for layer in layerList:
            a, b, c, d = layer['position']
            z = layer['depth']
            q, w = layer['textureCoordinate']
            p1 = np.array([a, b, z, 1, 0, 0])
            p2 = np.array([a, d, z, 1, w, 0])
            p3 = np.array([c, d, z, 1, w, q])
            p4 = np.array([c, b, z, 1, 0, q])
            model = matrix.scale(2 / psd_size[0], 2 / psd_size[1], 1) @ \
                matrix.translate(-1, -1, 0) @ \
                matrix.rotate_ax(-math.pi / 2, axis=(0, 1))
            glBindTexture(GL_TEXTURE_2D, layer['textureIndex'])
            glColor4f(1, 1, 1, 1)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBegin(GL_QUADS)
            for p in [p1, p2, p3, p4]:
                a = p[:4]
                b = p[4:6]
                a = a @ model
                a[0:2] *= a[2]
                h_rotate = 0
                if not layer['name'] == 'body':
                    h_rotate = math.sin(time.time() * 5) / 30
                a = a @ matrix.translate(0, 0, -1) \
                      @ matrix.rotate_ax(h_rotate, axis=(0, 2)) \
                      @ matrix.translate(0, 0, 1)
                # a = a @ matrix.scale(1,1,3) \
                #       @ matrix.rotate_ax(1.2, axis=(0, 2)) \
                #       @ matrix.translate(2.1, 0, 0.8)
                a = a @ matrix.perspective(999)
                glTexCoord2f(*b)
                glVertex4f(*a)
            glEnd()
        glfw.swap_buffers(window)


def addDepthInfo(layerList):
    with open('F:/Vtuber_Genshin/depth.yaml', encoding='utf8') as f:
        depthInfo = yaml.load(f,Loader=yaml.FullLoader)
    for layer in layerList:
        if layer['name'] in depthInfo:
            layer['depth'] = depthInfo[layer['name']]


if __name__ == '__main__':
    psd = PSDImage.open('F:/Vtuber_Genshin/moon_rabit/limo_simple.psd')
    layerList, size = extractLayers(psd)
    addDepthInfo(layerList)
    for layer in layerList:
        print(layer['name'])
    openglDrawLoop(layerList, size)
