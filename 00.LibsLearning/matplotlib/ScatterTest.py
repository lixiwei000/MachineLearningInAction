import matplotlib.pyplot as plt
from math import *
x = range(1,20)
y = [pow(t,2) for t in x]
z = [sqrt(t) for t in x]
k = [log2(t) for t in x]
fig = plt.figure()
ax1 = fig.add_subplot(221)           # 2*1   no.2
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


ax1.scatter(x,x)
ax2.scatter(x,y)
ax3.scatter(x,z)
ax4.scatter(x,k)
plt.show()


