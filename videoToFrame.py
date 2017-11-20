import os
import subprocess as sp


def vtf(videoLoc,outputLoc):
	cmd='ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(videoLoc)
	dur=sp.check_output(cmd,shell=True)
	dur=16/float(dur)
	cmd='ffmpeg -i {} -vf fps={} {}out%d.jpg'.format(videoLoc,str(dur),outputLoc)
	sp.call(cmd,shell=True)



#testData
file=open('testNames.txt','r')
for line in file:
	subpath=line[:-5]
	testpath="TestData/"+subpath+"/"
	inputt="Data/"+line[:-1]
	if not os.path.exists(testpath):
		os.makedirs(testpath)
	vtf(inputt,testpath)
file.close()

#trainData
file=open('trainNames.txt','r')
for line in file:
	t=7
	if(line[-t]!='.'):
		t=t+1
	subpath=line[:-t]
	testpath="TrainData/"+subpath+"/"
	# print(testpath)
	inputt="Data/"+line[:-3]
	if not os.path.exists(testpath):
		os.makedirs(testpath)
	vtf(inputt,testpath)
file.close()
