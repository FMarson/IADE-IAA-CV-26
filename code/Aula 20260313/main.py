import PIL.Image
import numpy
import random

if __name__ == '__main__':
    # Preencher uma  imagem com valores random
    image1 = numpy.random.randint(low = 0, high = 256, size=(800,600,3))
    #print(image1)
    picture1 = PIL.Image.fromarray(image1.astype('uint8'), 'RGB')
    picture1.save('image1.png')

    # Preencher com preto
    image2 = numpy.zeros(shape=(600,800,3))
    #print(image2)
    picture2 = PIL.Image.fromarray(image2.astype('uint8'), 'RGB')
    picture2.save('image2.png')

    image3 = 128 * numpy.ones(shape=(800,600,3))
    #print(image3)
    picture3 = PIL.Image.fromarray(image3.astype('uint8'), 'RGB')
    picture3.save('image3.png')

    #com transparência
    #image4 = 128 * numpy.ones(shape=(800,600,4))
    #print(image4)
    #picture4 = PIL.Image.fromarray(image4.astype('uint8'), 'RGBA')
    #picture4.save('image4.png')

    #signed = - +
    #unsigned = + 

    #Lembrar que as matrizes tem ordem diferente da imagem:
    #Matriz l, c
    #Imagem x, y

    image5 = PIL.Image.open("./tela_login.png")
    image5 = numpy.array(image5)
    print(image5.shape)

    for x_pixel in range (741):
        for y_pixel in range (1067):
            image5[y_pixel][x_pixel][0] = 0
            image5[y_pixel][x_pixel][2] = 0

    picture5 = PIL.Image.fromarray(image5.astype('uint8'), 'RGBA')
    picture5.save('nova.png')
    