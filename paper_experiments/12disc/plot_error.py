import matplotlib.pyplot as plt
import seaborn
import numpy as np

def moving_average(x,N):
	cs = np.cumsum(x)
	return (cs[N:]-cs[:-N])/float(N)

data = np.loadtxt("dense4_error.txt")
data2 = np.loadtxt("denseskip4_error.txt")
data3 = np.loadtxt("denseskip8_error.txt")
data4 = np.loadtxt("error_shallow_perm3.txt")
data5 = np.loadtxt("error_perm3.txt")
data6 = np.loadtxt("error_permskip3.txt")
data7 = np.loadtxt("error_permskip3max.txt")

#~ data5 = np.loadtxt("dense8_error.txt")

plt.plot(moving_average(data[:,1], 100), label="Dense-4")
plt.plot(moving_average(data2[:,1], 100), label="Dense-Skip-4")
plt.plot(moving_average(data3[:,1], 100), label="Dense-Skip-8")
plt.plot(moving_average(data4[:,1], 100), label="Perm-3,1")
plt.plot(moving_average(data5[:,1], 100), label="Perm-3,4")
plt.plot(moving_average(data6[:,1], 100), label="Perm-Skip-3,4")
plt.plot(moving_average(data7[:,1], 100), label="Perm-Skip-3,4-Max")

plt.xlabel("Epoch")
plt.ylabel("MSE (running avg)")
plt.legend()
plt.xlim(0,20000)
plt.show()

print "Dense-4: ", np.mean(data[10000:,1])
print "Dense-Skip-4: ", np.mean(data2[10000:,1])
print "Dense-Skip-8: ",np.mean(data3[10000:,1])
print "Perm-3,1: ",np.mean(data4[10000:,1])
print "Perm-3,4: ",np.mean(data5[10000:,1])
print "Perm-Skip-3,4: ",np.mean(data6[10000:,1])
print "Perm-Skip-3,4-Max: ",np.mean(data7[10000:,1])
