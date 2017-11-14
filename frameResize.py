import os
from PIL import Image
import numpy

file=open('testNames.txt','r')
for line in file:
	subpath=line[:-5]
	testpath="TestData/"+subpath+"/"
	name="out{}.jpg"
	i = 1
	while os.path.isfile(testpath+name.format(str(i))):
		# print(testpath+name.format(str(i)))
		im=Image.open(testpath+name.format(str(i)))
		im=im.resize((112,112))
		pix = numpy.array(im.getdata()).reshape(112, 112, 3)
		# im.save(testpath+name.format(str(i))[:-4],'JPEG')
		if not os.path.exists("Resized/"+testpath):
			os.makedirs("Resized/"+testpath)
		im.save("Resized/"+testpath+name.format(str(i)),'JPEG')
		# print(pix.shape)
		i=i+1
		# print(x.shape)
file.close()

t=7
file=open('trainNames.txt','r')
for line in file:
	if(line[-t]=='.'):
		t=t+1
	subpath=line[:-t]
	testpath="TrainData/"+subpath+"/"
	name="out{}.jpg"
	i = 1
	while os.path.isfile(testpath+name.format(str(i))):
		# print(testpath+name.format(str(i)))
		im=Image.open(testpath+name.format(str(i)))
		im=im.resize((112,112))
		pix = numpy.array(im.getdata()).reshape(112, 112, 3)
		# im.save(testpath+name.format(str(i))[:-4],'JPEG')
		if not os.path.exists("Resized/"+testpath):
			os.makedirs("Resized/"+testpath)
		im.save("Resized/"+testpath+name.format(str(i)),'JPEG')
		# print(pix.shape)
		i=i+1
		# print(x.shape)
file.close()


