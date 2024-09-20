import cv2
import matplotlib.pyplot as plt 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
img = cv2.imread('img.jpg')
img = cv2.resize(img, (1000,650))
# convert to RGB
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# plt.show()

# convert to Gray Scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray, cmap='gray')
# plt.show()

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(80,80))
for (x, y, h, w) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("Face",img)
cv2.waitKey(0)