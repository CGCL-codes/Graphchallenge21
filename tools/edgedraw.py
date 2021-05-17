import matplotlib.pyplot as plt

def hashElement(v,buckets,max_range):
    if v>= max_range:
        return v
    else:
        return v/buckets+v%buckets*(max_range/buckets)
    pass

num = str(1);
neuron = str(4096);
bucketnumber = 256;

path='../src/tmp.txt'

a = []
b = []
file =open(path,'r')
for eachline in file.readlines():
    x=eachline.split('\t')
    a.append(hashElement(int(x[1])-1,bucketnumber,int(neuron)))#得到列的id
    b.append(hashElement(int(x[0])-1,bucketnumber,int(neuron)))#得到行的id
    #a.append(int(x[1])-1)#得到列的id
    #b.append(int(x[0])-1)#得到行的id
plt.title("neuron:"+neuron+" bucket"+str(bucketnumber)+" layer:"+num)
plt.xlim(xmax=64,xmin=0)
plt.ylim(ymax=64,ymin=0)
plt.xlabel("col")
plt.ylabel("row")
plt.plot(a,b,'.')

plt.savefig("tmp.fig")

# plt.show()
