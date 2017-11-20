# for testNames file

# file =open('testNames.txt','r')
# filew = open('yo','w')
# a = [0,45,86,129,164,200,236,269,303,342,372,415,-1]
# i=1
# j=1
# for line in file:
# 	# print(i,a[j])
# 	if(i==a[j]):
# 		j=j+1
# 	x=line[:-1]+" "+str(j)+line[-1]
# 	filew.write(x)
# 	i=i+1
# file.close()
# filew.close()

################################################################################################

# for trainNames file

file =open('trainNames.txt','r')
filesp = [open("TrainFiles/trainNames{}.txt".format(x),'w') for x in range(1,6)]
a = [0,102,206,313,412,507,609,716,793,885,956,1074,-1]
i=0
j=1
t=3
for line in file:
	if(line[-t]!=' '):
		t=t+1
	# print(i,a[j])
	if(i==a[j]):
		j=j+1
	x=line[:-t]+" "+str(j)+line[-1]
	gno=line[-(t+10):-(t+8)]
	gno=int(gno)
	# print(x,gno,t)
	filesp[gno%5].write(x)
	i=i+1
file.close()
for x in range(0,5):
	filesp[x].close()