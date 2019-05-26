import numpy as np
from random import randrange
from PIL import Image

def generate_random_color_image(image_shape=(40,40,3)):
    rnd_colors = np.zeros(image_shape, dtype='uint8')
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            for k in range(3):
                    v = randrange(0, 256)
                    rnd_colors[i][j][k] = v
    return rnd_colors

def neighborhood_func(dist, epoch):
    sigma = 255 * 4 / ((epoch + 1) ** 2)
    value = np.exp(-(dist * dist) / (2 * sigma))
    return value

def kohonen(dataset, map_shape=(40, 40, 3), epochs=3, learning_rate=0.8):
    nodes = np.random.randint(256, size=map_shape).astype(float)
    # nodes = np.zeros(map_shape).astype(float)
    dateshpae = dataset.shape
    n = [(i, j) for i in range(map_shape[0]) for j in range(map_shape[1])]
    for e in range(epochs):
        print('epoch:', e+1)
        for i in range(dateshpae[0]):
            for j in range(dateshpae[1]):
                # find nearest
                row = 0
                col = 0
                min_value = np.linalg.norm(dataset[i][j] - nodes[row][col])
                for r in range(map_shape[0]):
                    for c in range(map_shape[1]):
                        dist = np.linalg.norm(dataset[i][j] - nodes[r][c])
                        if dist < min_value:
                            min_value = dist
                            row = r
                            col = c
                winner = (row, col)
                # update weights
                for node in n:
                    # dist = np.linalg.norm(nodes[winner[0]][winner[1]] - nodes[node[0]][node[1]])
                    dist = ((winner[0] - node[0]) ** 2 + (winner[1] - node[1]) ** 2) ** 0.5
                    delta = learning_rate \
                            * neighborhood_func(dist, e) \
                            * (dataset[i][j] - nodes[node[0]][node[1]])
                    new_value = nodes[node[0]][node[1]] + delta
                    nodes[node[0]][node[1]] = new_value
    return nodes.astype('uint8')


def main():
    dataset = generate_random_color_image()
    img = Image.fromarray(dataset, 'RGB')
    img.save('random.png')
    sofm = kohonen(dataset)
    sofm_img = Image.fromarray(sofm, 'RGB')
    sofm_img.save('SOFM.png')


main()
