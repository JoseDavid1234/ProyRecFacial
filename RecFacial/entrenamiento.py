import cv2
import os
import numpy as np

dataPath = 'D:/Programacion/CarpPython/ProyRecFacial/DataFacial'
peopleList = os.listdir(dataPath)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir

	for fileName in os.listdir(personPath):
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
	label = label + 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modeloLBPHFace.xml')