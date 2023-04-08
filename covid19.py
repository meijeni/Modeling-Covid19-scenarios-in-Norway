#Problem 1.4. Simulate Covid19 in Norway
from SEIR_interaction import *
import numpy as np
import matplotlib.pyplot as plt

#a)
def extract_data(filename):
    infile = open(filename, "r")
    list = []
    for line in infile:
        words = line.split("/n")
        for word in words:
            word = word.split(";")
            name = (word[1].strip())
            S_0 = (float(word[2].strip()))
            E2_0 = (float(word[3].strip()))
            lat = (float(word[4].strip()))
            long = (float(word[5].strip()))
            region = RegionInteraction(name, S_0, E2_0, lat, long)
            list.append(region)
    infile.close()
    return list
a = extract_data("fylker.txt")


#b)
def covid19_Norway(beta, filename, num_days, dt):
    data = extract_data(filename)
    problem = ProblemInteraction(data, filename, beta)
    solver = SolverSEIR(problem, num_days, dt)
    problem.set_initial_condition()
    solver.solve()
    plt.figure(figsize=(9, 12))
    index = 1
    for i in data:
        plt.subplot(4,3,index)
        i.plot()
        index += 1
    plt.subplot(4,3,index)
    plt.subplots_adjust(hspace = 0.75, wspace=0.5)
    problem.plot()
    plt.legend()
    plt.show()

covid19_Norway(0.5, "fylker.txt", 300, 1.0)

"""
Terminal> python covid19.py
"""


#c)
#calculating values
from datetime import date
d0 = date(2020, 2, 15)
d1 = date(2020, 3, 14)
delta = d1 - d0
print(delta.days)
d1 = date(2020, 4, 20)
delta = d1 - d0
print(delta.days)
d1 = date(2020, 5, 10)
delta = d1 - d0
print(delta.days)
d1 = date(2020, 6, 30)
delta = d1 - d0
print(delta.days)
d1 = date(2020, 7, 31)
delta = d1 - d0
print(delta.days)
d1 = date(2020, 8, 30)
delta = d1 - d0
print(delta.days)
d1 = date(2020, 11, 22)
delta = d1 - d0
print(delta.days)

def values(R):
    r_ia = 0.1; r_e2=1.25; lmbda_1=0.33; lmbda_2=0.5; p_a=0.4; mu=0.2
    return R/((r_e2/lmbda_2 + r_ia/mu + 1/mu))
def beta(t):
    if t < 28:
        return values(4.0)
    if 28 <= t < 66:
        return values(0.5)
    if 66 <= t < 86:
        return values(0.4)
    if 86 <= t < 137:
        return values(0.8)
    if 137 <= t < 168:
        return values(0.9)
    if 168 <= t < 198:
        return values(1.0)
    if 198 <= t < 282:
        return values(4.0)

t_list = []
b = []
for i in range(283):
    t_list.append(i)
    b.append(beta(i))
b = np.array(b)
t = np.array(t_list)
plt.plot(t,b)
plt.xlabel("t")
plt.ylabel("beta(t)")
plt.show()

"""
Terminal> python covid19.py
28
65
85
136
167
197
281
"""
