import cv2
import numpy as np
import math

def preprocess(imgOriginal):

    imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

    imgBlurred = cv2.GaussianBlur(imgGrayscale, (3, 3), 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12)

    return imgGrayscale, imgThresh

def preprocessPlate(imgOriginal):

    imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

    imgBlurred = cv2.GaussianBlur(imgGrayscale, (3, 3), 0)

    imgThresh = cv2.adaptiveThreshold(imgGrayscale, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15 , 11)

    return imgGrayscale, imgThresh










