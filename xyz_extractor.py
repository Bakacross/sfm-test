f = open('pos.txt','r')
fout = open('xyz.csv','w')
#fout.write('x,y,z\n')
hihi = False
for line in f:
    fout.write(line[6:len(line)-2]+'\n')
f.close()
fout.close()
