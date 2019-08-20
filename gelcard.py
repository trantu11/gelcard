import numpy as np
import cv2 

img = cv2.imread('C:/Users/PV/Desktop/gelcard/data/gelcard2.jpg')

# roi = img[800:800+700,0:0+3024]
roi = img[350:350+500,0:0+2102]
roi = cv2.resize(roi,None,fx=0.5,fy=0.5)
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

lower_red = np.array([0,69,57])
upper_red = np.array([18,255,255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([340,69,57])
upper_red = np.array([360,255,255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

mask = mask1+mask2
output = cv2.bitwise_and(hsv, roi, mask = mask)

gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)

ret, threshold = cv2.threshold(gray, 10, 255, 0)

kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(threshold,kernel,iterations = 1)

contours, hierarchy =  cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

total_contours = len(contours)
print('Phat hien: ',total_contours)

locations = []
for contour in contours:
	M = cv2.moments(contour)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	print('Tam theo truc x:',cX,' Tam theo truc y',cY)
	locations.append([cX,cY])

def takeFirst(elem):
    return elem[0]

locations.sort(key=takeFirst)

stringState = []

for location in locations:
	indexY = location[1]
	indexX = location[0]
	if indexY > 150:
		cv2.circle(roi,(indexX,indexY), 3, (255,255,255), -1)
		stringState.append('Thap')
	else:
		cv2.circle(roi,(indexX,indexY), 3, (0,0,0), -1)
		stringState.append('Cao')

print(locations)
print(stringState)

cv2.imshow('roi',roi)
cv2.imshow('output',output)
cv2.imshow('dilation',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()