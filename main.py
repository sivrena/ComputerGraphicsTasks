import cv2
import argparse
import numpy
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"], cv2.IMREAD_COLOR)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#вывести исходное изображение
cv2.imshow("Image", image)
cv2.waitKey(0)

#grayscale изображение
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("Grayscale image", gray)
cv2.waitKey(0)

#hsv изображение
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
cv2.imshow("hsv image", hsv)
cv2.waitKey(0)

#отраженное по нижней границе изображение
mirror = cv2.flip(img, 0)
cv2.imshow("Mirrored image", mirror)
cv2.waitKey(0)

#повернутое на 45 градусов
angle = 45 * numpy.pi / 180
rot_mx = numpy.array([[numpy.cos(angle), -numpy.sin(angle), 0],
      [numpy.sin(angle), numpy.cos(angle), 0]])
rotated = cv2.warpAffine(img, rot_mx, dsize=(img.shape[1], img.shape[0]))
cv2.imshow("Rotated image", rotated)
cv2.waitKey(0)

#смещённое на 10 пикселей вправо
translation = numpy.asarray([10, 0]).reshape(2, 1)
mx = numpy.concatenate((numpy.identity(2), translation), axis=1)
shifted = cv2.warpAffine(img, mx, dsize=(img.shape[1], img.shape[0]))
cv2.imshow("Shifted image", shifted)
cv2.waitKey(0)

#изменённая яркость
brightened = cv2.convertScaleAbs(img, alpha = 1, beta=0)
cv2.imshow("Brightened image", brightened)
cv2.waitKey(0)

#изменённый контраст
contrasted = cv2.convertScaleAbs(img, alpha = 2, beta=0)
cv2.imshow("Contrasted image", contrasted)
cv2.waitKey(0)

#бинаризация изображения
#img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary image", binary)
cv2.waitKey(0)

#размытие изображения
blurred = cv2.GaussianBlur(image, (51, 51), 0)
cv2.imshow("Binarized image", blurred)
cv2.waitKey(0)

#эрозия к изображению
erosion = cv2.erode(img, numpy.ones((5,5), numpy.uint8), iterations = 1)
cv2.imshow("Erosed image", erosion)
cv2.waitKey(0)