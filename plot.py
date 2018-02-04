import numpy as np
import matplotlib.pyplot as plt
import math

data1 = np.transpose(np.loadtxt("initial.tsv"))
data2 = np.transpose(np.loadtxt("solution.tsv"))
data3 = np.transpose(np.loadtxt("analyt.tsv"))

positions = np.linspace(0,1,data1.shape[0])

plt.plot(data1)
plt.show()
plt.plot(data2)
plt.plot(data3)
plt.show()

error = np.log10(np.fabs((data2-data3)/data3))
plt.plot(positions,error)
plt.ylabel("Logarithmic Error")
plt.xlabel("Position")
plt.tight_layout()
plt.savefig("error.pdf")
plt.show()