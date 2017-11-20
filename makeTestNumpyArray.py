import os
from PIL import Image
import numpy

#######################################################################
file=open('testFile','r')
x=[]
xx = []
for line in file:
	t=7
	tt=2
	if(line[-t]!='.'):
		t=t+1
		tt=tt+1
	subpath=line[:-t]
	room = int(line[-tt:-1])
	testpath="Resized/TestData/"+subpath+"/"
	# print(t,line,subpath)
	i=1
	y=[]
	name="out{}.jpg"
	while os.path.isfile(testpath+name.format(str(i))):
		# print(testpath+name.format(str(i)))
		im=Image.open(testpath+name.format(str(i)))
		# im=im.resize((112,112))
		pix = numpy.array(im.getdata()).reshape(112, 112, 3)
		y.append(pix)
		i=i+1
		if(i==17):
			break
		# print(i)
	x.append(y)
	xx.append(room)
print('starting to save')
xdata=numpy.array(x)
ydata=numpy.array(xx)
print (xdata.shape)
print (ydata.shape)
numpy.save('testX', xdata)
numpy.save('testY', ydata)
file.close()
#######################################################################