import cv2
import os
from openpyxl import Workbook
import time

asistentes = {}

dataPath = 'D:/Programacion/CarpPython/ProyRecFacial/DataFacial' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture('rtsp://admin:parlanchin123@192.168.0.108:80/cam/realmonitor?channel=1&subtype=0&unicast=true')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()
	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
		result = face_recognizer.predict(rostro)

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
		if result[1] < 70:
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			t=time.localtime()
			asistentes[imagePaths[result[0]]] = ''+time.strftime("%H:%M:%S",t)
		else:
			cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		
	cv2.imshow('frame',frame)
	k = cv2.waitKey(1)
	if k == 27:
		break

book = Workbook()
sheet = book.active

sheet['A1']='Alumno'
sheet['B1']='Hora de ingreso'

puntero = 2
for key in asistentes:
	sheet[f'A{puntero}']=key
	sheet[f'B{puntero}']=asistentes[key]
	puntero+=1

book.save('pruebaExcel.xlsx')

cap.release()
cv2.destroyAllWindows()