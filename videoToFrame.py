import os
import subprocess as sp

file=open('testNames.txt','r')

for line in file:
	subpath=line[:-5]
	testpath="TestData/"+subpath+"/"
	inputt="Data/"+line[:-1]
	if not os.path.exists(testpath):
		os.makedirs(testpath)
	cmd='ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(inputt)
	dur=sp.check_output(cmd,shell=True)
	# print("dur - ",dur)
	dur=16/float(dur)
	# print("dur - ",dur)
	cmd='ffmpeg -i {} -vf fps={} {}out%d.jpg'.format(inputt,str(dur),testpath)
	# cmd='ffmpeg -i {} {}out%d.jpg'.format(inputt,testpath)
	# print (cmd)
	sp.call(cmd,shell=True)


file.close()

file=open('trainNames.txt','r')

for line in file:
	subpath=line[:-7]
	testpath="TrainData/"+subpath+"/"
	inputt="Data/"+line[:-3]
	if not os.path.exists(testpath):
		os.makedirs(testpath)
	cmd='ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(inputt)
	dur=sp.check_output(cmd,shell=True)
	# print("dur - ",dur)
	dur=16/float(dur)
	# print("dur - ",dur)
	cmd='ffmpeg -i {} -vf fps={} {}out%d.jpg'.format(inputt,str(dur),testpath)
	# cmd='ffmpeg -i {} {}out%d.jpg'.format(inputt,testpath)
	# print (cmd)
	sp.call(cmd,shell=True)


file.close()
