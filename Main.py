import cv2
import numpy as np
import os
import math
import requests

import DetectChars
import DetectPlates
import PossiblePlate
from tkinter.filedialog import askopenfilename

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
showSteps = True   

def main():
    z = 1
    DetectChars.loadKNNDataAndTrainKNN()
    filename = askopenfilename()
    cap = cv2.VideoCapture(filename) 
    recognizedPlates = []
    while True:
        ret, frame = cap.read()
        if (ret):
            listOfPossiblePlates = DetectPlates.detectPlatesInScene(frame)

            listOfPossiblePlates, recognizedPlate, recognizedPlates = DetectChars.detectCharsInPlates(listOfPossiblePlates, recognizedPlates)
            print(recognizedPlates)
            print(recognizedPlate)
            if len(listOfPossiblePlates) == 0 or recognizedPlate == True:                         
                continue
            else:                                                      
                listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
                licPlate = listOfPossiblePlates[0]
                cv2.imwrite("plates/imgPlate" + str(z) + ".png", licPlate.imgPlate)
                cv2.imwrite("plates/imgTresh" + str(z) + ".png", licPlate.imgThresh)

                if len(licPlate.strChars) == 0:                    
                    continue                                         
    
                print("\nPlate  = " + licPlate.strChars + "\n") 
                print("----------------------------------------")

            z = z + 1
        else:
            break
         
    cap.release()
    cv2.waitKey(0)

    return

if __name__ == "__main__":
    main()


















