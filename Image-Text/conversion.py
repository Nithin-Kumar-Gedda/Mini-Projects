import matplotlib.pyplot as plt 
import PIL
import pytesseract
import re

img = PIL.Image.open('img.png')
plt.imshow(img)
# plt.show()

# convert img to text
pytesseract.pytesseract.tesseract_cmd =r'C:/Program Files/Tesseract-OCR/tesseract.exe'

text_data = pytesseract.image_to_string(img.convert('RGB'),lang='eng')
# print(text_data)

m=re.search("engine = (\w+)",text_data)
print(m[1])