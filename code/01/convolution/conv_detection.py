from PIL import Image
import numpy
from matplotlib import pyplot

KX_SIZE = 10
KY_SIZE = 10
DETECTION_THRESHOLD = 255

def kernel(image, kx, ky, kx_size = KX_SIZE, ky_size= KY_SIZE):

    #if numpy.sum(image[kx: kx + kx_size, ky: ky + ky_size, 0]) > 0:
    #    print(image[kx: kx + kx_size, ky: ky + ky_size, 0])

    convolution = numpy.sum(image[kx: kx + kx_size, ky: ky + ky_size, 0])
    convolution /= kx_size * ky_size

    return convolution

if __name__ == '__main__':

    picture = Image.open('square2.png')
    image = numpy.array(picture)

    x_size, y_size, n_channels = image.shape

    distance_matrix = numpy.zeros((x_size, y_size, n_channels))
    
    for x_pixel in range(x_size - KX_SIZE):
        for y_pixel in range(y_size - KY_SIZE):
            dd = kernel(image = image, kx = x_pixel, ky = y_pixel)

            for channel in range (n_channels):
                distance_matrix[x_pixel + int(KX_SIZE / 2)][y_pixel + int(KY_SIZE / 2)][channel] = dd
    
    picture = Image.fromarray(distance_matrix.astype('uint8'), 'RGB')
    picture.save('convolution.png')

    x_centers, y_centers = numpy.where(distance_matrix[:, :, 0] > DETECTION_THRESHOLD)

    for x_pixel in range (x_size):
        for y_pixel in range(y_size):

                if x_pixel in x_centers or y_pixel in y_centers:
                     image[x_pixel][y_pixel][0] = 255
                     image[x_pixel][y_pixel][1] = 0
                     image[x_pixel][y_pixel][2] = 0

    picture = Image.fromarray(image.astype('uint8'), 'RGB')
    picture.save('image_detected.png')