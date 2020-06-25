import cv2
import numpy as np 
import argparse
import time
from math import sqrt
import math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--web_cam', help="True/False", default=False)
parser.add_argument('--video', help="True/False", default=False)
parser.add_argument('--imagen', help="True/False", default=False)
parser.add_argument('--video_path', help="ruta del video", default="videos/sicales.mp4")
parser.add_argument('--image_path', help="ruta imagen a detectar", default="imagenes/m2.jpg")
parser.add_argument('--verbose', help="ayuda de la entrada de los argumentos", default=True)
args = parser.parse_args()

#descargue yolo de 608 pesos
#carga yolo
#Se deben descargar los pesos y el archivo cfg de https://pjreddie.com/darknet/yolo
# se descargar  names de https://github.com/pjreddie/darknet/blob/master/data/coco.names
def load_yolo():
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
    
	layers_names = net.getLayerNames() #devuelve 254 capas convolucionales 
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()] # Devuelve índices de capas con salidas no conectadas.
	colors = np.random.uniform(0, 255, size=(len(classes), 3)) #genera colores aleatorios por cada clase
	return net, classes, colors, output_layers

def cargar_imagen(img_path):
	# carga la imagen
	img = cv2.imread(img_path)#carga la imagen
	img = cv2.resize(img, None, fx=0.4, fy=0.4)#redimensiona la imagen 
	height, width, channels = img.shape
	return img, height, width, channels

def iniciar_webcam():
	cap = cv2.VideoCapture(0)

	return cap

#Crea blob de 4 dimensiones a partir de la imagen
def detect_objects(img, net, outputLayers):
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(608, 608), mean=(100, 50, 40), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)#retorna cada capa
	return blob, outputs

#se crea una lista llamada puntajes que almacena el porcentaje de deteccion

def obtener_cajas_dimensiones(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			#Seleccionamos todos los cuadros con los porcentajes de detenccion  mayor al 30%
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width )
				h = int(detect[3] * height )
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h]) ## agreagar x,y,w,h a boxes
				confs.append(float(conf))
				class_ids.append(class_id)

	return boxes, confs, class_ids

# calcula la distancia entre dos puntos
def distancia_puntos_caja(x,y, x1,y1):
       return sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

# ayuda a suprimir detecciones dobles en la imagen  cv2.dnn.NMSBoxes
def dibujar_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_COMPLEX_SMALL # tipo de letra para las etiquetas
	metros = "m"
    
	for i in range(len(boxes)):
		x1, y1, w1, h1= boxes[i-1]
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			if ((label == "person") or (label == "bird") or (label == "dog")):
    				
			        color = colors[i]
			        color_texto = [25,10,75]
			        cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)
			        m = int(w/2)
					#radio = int(h/2)
			        m1 = int(h/2)
			        center_x = int(x+w /2)  #+m
			        center_y = int(y+h /2)
			        center_x1 = int(x1+w1 /2)   #-m1
			        center_y1 = int(y1+h1 /2)
			        distancia = distancia_puntos_caja(center_x,center_y, center_x1, center_y1)
			        cv2.circle(img,(center_x ,center_y),2,color,2)
			        cv2.circle(img,(center_x, center_y),m,color,2)
			        cv2.putText(img, label, (x, y - 15), font, 1, color, 1)
			        distenmetros = distancia/39.37 #convertir los pixels a metros
					
			        distanciasolcial = int(distenmetros)
			        if distanciasolcial <  1.5: #menor a 1.5 metros el rectangulo de pinta de rojo
    				    cv2.rectangle(img, (x,y), (x+w, y+h), color_texto, 2)

			        distpunto = str(distanciasolcial )
			        #print(distpunto)
			        cv2.putText(img,distpunto + metros, (center_x-20, center_y+24 ), cv2.FONT_HERSHEY_PLAIN, 1, color, 1,5)
			        cv2.line(img, (int(center_x ), int(center_y )), (int(center_x1 ), int(center_y1 )),
			        color, 1)

	cv2.imshow("Salida", img)

def imagen_detectada(img_path): 
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = cargar_imagen(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = obtener_cajas_dimensiones(outputs, height, width)
	dibujar_labels(boxes, confs, colors, class_ids, classes, image)
	cv2.imwrite('distaciasocial.jpg',image)
	while True:
		key = cv2.waitKey(27) #  presionar ESc para salir
		if key == 27:
			break

def webcam_detectada():
	model, classes, colors, output_layers = load_yolo()
	cap = iniciar_webcam()
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = obtener_cajas_dimensiones(outputs, height, width)
		dibujar_labels(boxes, confs, colors, class_ids, classes, frame)
		cv2.imwrite('distaciasocial.jpg',frame)
		key = cv2.waitKey(27) #  presionar ESc para salir
		if key == 27:
			break
	cap.release()


def iniciar_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = obtener_cajas_dimensiones(outputs, height, width)
		dibujar_labels(boxes, confs, colors, class_ids, classes, frame)
		cv2.imwrite('videosocial.jpg',frame)
		key = cv2.waitKey(27) #  presionar ESc para salir
		if key == 27:
			break
	cap.release()



if __name__ == '__main__':
	web_cam = args.web_cam
	video = args.video
	imagen = args.imagen
	if web_cam:
		if args.verbose:
			print('*** Inicia  Web Cam detectar Objectos ***')
		webcam_detectada()
	if video:
		Ruta_Video = args.video_path
		if args.verbose:
			print('*** Abre el Video de la ruta *** '+Ruta_Video+" ")
		iniciar_video(Ruta_Video)
	if imagen:
		image_path = args.image_path
		if args.verbose:
			print("*** Abre la imagen de la ruta *** "+image_path+" .... ")
		imagen_detectada(image_path)


	cv2.destroyAllWindows()