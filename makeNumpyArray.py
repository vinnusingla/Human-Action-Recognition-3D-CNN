import os
from PIL import Image
import numpy

q='train'
z='Train'
t=7
#######################################################################
file=open(q+'Names.txt','r')
x=[]
for line in file:
	if(q=='train' and line[-t]=='.'):
		t=t+1
	subpath=line[:-t]
	testpath=z+"Data/"+subpath+"/"
	i=1
	y=[]
	name="out{}.jpg"
	while os.path.isfile(testpath+name.format(str(i))):
		im=Image.open('Resized/'+testpath+name.format(str(i)))
		# im=im.resize((112,112))
		pix = numpy.array(im.getdata()).reshape(112, 112, 3)
		y.append(pix)
		i=i+1
		if(i==17):
			break
		# print(i)
	x.append(y)
print('starting to save')
xdata=numpy.array(x)
print (xdata.shape)
numpy.save('np'+q+'Array', xdata)
file.close()
#######################################################################