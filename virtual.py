import yaml
import math
import numpy as np
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

from psd_tools import PSDImage
from rimo_utils import matrix

import reality

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


def addDepthInfo(layerList):
    with open('F:/Vtuber_Genshin/depth.yaml', encoding='utf8') as f:
        depthInfo = yaml.load(f,Loader=yaml.FullLoader)
    for layer in layerList:
        if layer['name'] in depthInfo:
            layer['depth'] = depthInfo[layer['name']]


bufferFeature = None
def featureBuffer():
    global bufferFeature
    bufferRatio = 0.8
    newFeature = reality.get_feature()
    if bufferFeature is None:
        bufferFeature = newFeature
    else:
        bufferFeature = bufferFeature * bufferRatio + newFeature * (1 - bufferRatio)
    return bufferFeature


def metaFuse():
    glfw.window_hint(glfw.DECORATED, False)
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, True)
    glfw.window_hint(glfw.FLOATING, True)


def openglDrawLoop(layerList, psd_size):
    def generateTexture(img):
        w, h = img.shape[:2]
        d = 2**int(max(math.log2(w), math.log2(h)) + 1)
        texture = np.zeros([d, d, 4], dtype=img.dtype)
        texture[:w, :h] = img
        return texture, (w / d, h / d)

    vtuber_size = 512, 512
    
    glfw.init()
    metaFuse()
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(*vtuber_size, 'Vtuber', None, None)
    glfw.make_context_current(window)
    monitor_size = glfw.get_video_mode(glfw.get_primary_monitor()).size
    glfw.set_window_pos(window, monitor_size.width - vtuber_size[0], monitor_size.height - vtuber_size[1])

    glViewport(0, 0, *vtuber_size)

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

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
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        h_rotate, v_rotate = featureBuffer()
        for layer in layerList:
            a, b, c, d = layer['position']
            z = layer['depth']
            if type(z) in [int, float]:
                z1, z2, z3, z4 = [z, z, z, z]
            else:
                [z1, z2], [z3, z4] = z
            q, w = layer['textureCoordinate']
            p1 = np.array([a, b, z1, 1, 0, 0, 0, z1])
            p2 = np.array([a, d, z2, 1, z2 * w, 0, 0, z2])
            p3 = np.array([c, d, z3, 1, z3 * w, z3 * q, 0, z3])
            p4 = np.array([c, b, z4, 1, 0, z4 * q, 0, z4])

            model = matrix.scale(2 / psd_size[0], 2 / psd_size[1], 1) @ \
                matrix.translate(-1, -1, 0) @ \
                matrix.rotate_ax(-math.pi / 2, axis=(0, 1))
            glBindTexture(GL_TEXTURE_2D, layer['textureIndex'])
            glColor4f(1, 1, 1, 1)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBegin(GL_QUADS)
            for p in [p1, p2, p3, p4]:
                a = p[:4]
                b = p[4:8]
                a = a @ model
                a[0:2] *= a[2]
                if not layer['name'][:2] == 'body':
                    a = a @ matrix.translate(0, 0, -1) \
                          @ matrix.rotate_ax(h_rotate, axis=(0, 2)) \
                          @ matrix.rotate_ax(v_rotate, axis=(2, 1)) \
                          @ matrix.translate(0, 0, 1)
                a = a @ matrix.perspective(999)
                glTexCoord4f(*b)
                glVertex4f(*a)
            glEnd()
        glfw.swap_buffers(window)
    

if __name__ == '__main__':
    psd = PSDImage.open('F:/Vtuber_Genshin/moon_rabit/limo_simple.psd')
    layerList, size = extractLayers(psd)
    addDepthInfo(layerList)
    openglDrawLoop(layerList, size)