import os
from PIL import Image
import numpy

t=7
#######################################################################
#trainNames1.txt == VX , VY
#trainNames2.txt == X1 , Y1
#trainNames3.txt == X2 , Y2
#trainNames4.txt == X3 , Y3
#trainNames5.txt == X4 , Y4
no=1
file=open('TrainFiles/trainNames5.txt','r')
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
	# print(room)
	testpath="Resized/TrainData/"+subpath+"/"
	# print(t,line,subpath)
	i=1
	y=[]
	name="out{}.jpg"
	while os.path.isfile(testpath+name.format(str(i))):
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
numpy.save('X4', xdata)
numpy.save('Y4', ydata)
file.close()
#######################################################################