from skimage import feature
import matplotlib.pyplot as plt
import numpy as np
import cv2

method_lbp=['default',
			'ror',
			'uniform',
			'var']

cap=cv2.VideoCapture(0) # récupérer l'image de la caméra
numPoints=24
radius=3
id_method_lbp=0

while True:
	ret, frame=cap.read() # récupérer l'image

	image=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # transformer l'image en noir et blanc
    # faire appel à la fonction local_binary_pattern
	lbp=feature.local_binary_pattern(image, numPoints, radius, method=method_lbp[id_method_lbp])

	cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (100, 100, 100), cv2.FILLED)
	txt="[q] Quit  [o|l]numPoints:{:d}  [i|k]rayon:{:d} [m] Methode: {}".format(numPoints, radius, method_lbp[id_method_lbp])
	cv2.putText(frame, txt, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	cv2.imshow("Image", frame) # regarder la frame
	cv2.imshow("LBP", lbp/np.max(lbp)) # afficher l'image résultante

    # modifier les paramètres de manière plus simple et efficace
	key=cv2.waitKey(1)&0xFF
	if key==ord('m'):
		id_method_lbp=(id_method_lbp+1)%len(method_lbp)
	if key==ord('i'):
		radius=radius+1
	if key==ord('k'):
		radius=max(3, radius-1)
	if key==ord('o'):
		numPoints=numPoints+1
	if key==ord('l'):
		numPoints=max(3, numPoints-1)
	if key==ord('q'):
		quit()