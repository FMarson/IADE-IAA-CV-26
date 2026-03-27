from PIL import Image
import numpy
from matplotlib import pyplot

KX_SIZE = 10
KY_SIZE = 10
DETECTION_THRESHOLD = 245

if __name__ == '__main__':
    picture = Image.open('square.png')
    image = numpy.array(picture)

    x_size, y_size, n_channels = image.shape
    r_image = numpy.array(image[:,:,0])
    r_kernel = numpy.ones((KX_SIZE, KY_SIZE))

    tensor_product = numpy.tensordot(r_kernel, r_image, axes = 0)
    
    distance_matrix = numpy.sum(tensor_product, axis=(0,1))
    distance_matrix /= KX_SIZE * KY_SIZE
    
    distance_matrix = numpy.array([distance_matrix, distance_matrix, distance_matrix])
    distance_matrix = numpy.transpose(distance_matrix, axes=(1, 2, 0))
    print(distance_matrix.shape)

    x_centers, y_centers = numpy.where(distance_matrix[:, :, 0] > DETECTION_THRESHOLD)

    picture = Image.fromarray(distance_matrix.astype('uint8'), 'RGB')
    picture.save('tensor_product.png')

    for x_pixel in range (x_size):
        for y_pixel in range(y_size):
                if x_pixel in x_centers or y_pixel in y_centers:
                     distance_matrix[x_pixel][y_pixel][0] = 255
                     distance_matrix[x_pixel][y_pixel][1] = 0
                     distance_matrix[x_pixel][y_pixel][2] = 0
    
    picture = Image.fromarray(distance_matrix.astype('uint8'), 'RGB')
    picture.save('detected_tensor.png')
