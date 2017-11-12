import os
from PIL import Image
import numpy

file=open('testNames.txt','r')
x=[]
for line in file:
	subpath=line[:-5]
	testpath="TestData/"+subpath+"/"
	i=1
	y=[]
	name="out{}.jpg"
	while os.path.isfile(testpath+name.format(str(i))):
		# print(testpath+name.format(str(i)))
		im=Image.open(testpath+name.format(str(i)))
		im=im.resize((128,128))
		pix = numpy.array(im.getdata()).reshape(128, 128, 3)
		# im.save(testpath+name.format(str(i))[:-4],'JPEG')
		# print(pix.shape)
		y.append(pix)
		# print(x.shape)
		i=i+1
		if(i==17):
			break
		# print(i)
	x.append(y)
xdata=numpy.array(x)
print (xdata.shape)
file.close()
