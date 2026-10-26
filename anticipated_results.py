'''
Anticipated results once training is implemented 
Authors: Kiana Tarter & Francisco Herrera

'''
import matplotlib.pyplot as plt

# Choose network sizes in multiples of 4
sizes = [4, 8, 16, 32, 64, 128, 256]

# Made up accuracies based 
point  = [70, 78, 85, 91, 94, 96, 97]   
linear = [75, 82, 89, 92, 94, 96, 97]   
active = [85, 88, 91, 93, 95, 97, 98]  


plt.plot(sizes, point,  marker="o", label="Point neurons (LIF)")
plt.plot(sizes, linear, marker="o", label="Linear dendrites")
plt.plot(sizes, active, marker="o", label="Active dendrites (ours)")


plt.xscale("log", base=2)
plt.xticks(sizes, sizes)
plt.ylim(60, 100)
plt.xlabel("Number of hidden neurons")
plt.ylabel("Test Accuracy (%)")
plt.title("Anticipated: Accuracy vs. Network Size on Yin Yang Dataset")
plt.legend()

plt.savefig("anticipated_results.png", dpi=200)
plt.show()
