#Problem 1.3. The SEIR model across regions
from SEIR import *
import numpy as np
from ODESolver import *
import matplotlib.pyplot as plt

#a)
class RegionInteraction(Region):
    def __init__(self, name, S_0, E2_0, lat, long):
        self.latitude = lat*(np.pi/180)
        self.longitude = long*(np.pi/180)
        super().__init__(name, S_0, E2_0)
    def distance(self, other):
        R_earth = 64
        a = np.arccos(np.sin(self.latitude)*np.sin(other.latitude) + np.cos(self.latitude)*np.cos(other.latitude)*np.cos(abs(self.longitude - other.longitude)))
        if 0<=a<=1:
            pass
        else:
            print("Not passed if-test!")
        return R_earth*(np.arccos(np.sin(self.latitude)*np.sin(other.latitude) + np.cos(self.latitude)*np.cos(other.latitude)*np.cos(abs(self.longitude - other.longitude))))

#b)
class ProblemInteraction(ProblemSEIR):
    def __init__(self, region, area_name, beta, r_ia = 0.1, r_e2=1.25,lmbda_1=0.33, lmbda_2=0.5, p_a=0.4, mu=0.2):
        super().__init__(region, beta, r_ia=0.1, r_e2=1.25, lmbda_1=0.33, lmbda_2=0.5, p_a=0.4, mu=0.2)
        self.area_name = area_name
    def get_population(self):
        s = 0
        for i in range(len(self.region)):
            s += self.region[i].population
        return s
    def set_initial_condition(self):
        self.initial_condition = []
        for i in range(len(self.region)):
            ic = [self.region[i].S0, self.region[i].E1_0, self.region[i].E2_0, self.region[i].I0, self.region[i].Ia_0, self.region[i].R_0]
            self.initial_condition += ic
        return self.initial_condition
    def __call__(self, u, t):
        n = len(self.region)
        SEIR_list = [u[i:i+6] for i in range(0, len(u), 6)]
        E2_list = [u[i] for i in range(2, len(u), 6)]
        Ia_list = [u[i] for i in range(4, len(u), 6)]
        derivative = []
        self.N = sum(u)
        for i in range(n):
            S, E1, E2, I, Ia, R = SEIR_list[i]
            dS = 0
            N_i = self.region[i].population
            for j in range(n):
                E2_other = E2_list[j]
                Ia_other = Ia_list[j]
                N_j = self.region[j].population
                dij = self.region[i].distance(self.region[j])
                dS += -self.beta(t)*S*I/self.N - self.r_ia*self.beta(t)*S*Ia_other/N_j - self.r_e2*self.beta(t)*S*(E2_other/N_j)*np.exp(-dij)
            dE1 = - dS - self.lmbda_1*E1
            dE2 = self.lmbda_1*(1-self.p_a)*E1 - self.lmbda_2*E2
            dI = self.lmbda_2*E2 - self.mu*I
            dIa = self.lmbda_1*self.p_a*E1 - self.mu*Ia
            dR = self.mu*(I + Ia)
            derivative += [dS, dE1, dE2, dI, dIa, dR]
        return derivative

    def solution(self, u, t):
        n = len(t)
        n_reg = len(self.region)
        self.t = t
        self.S = np.zeros(n)
        self.E1 = np.zeros(n)
        self.E2 = np.zeros(n)
        self.I = np.zeros(n)
        self.Ia = np.zeros(n)
        self.R = np.zeros(n)
        SEIR_list = [u[:, i:i+6] for i in range(0, n_reg*6, 6)]
        for part, SEIR in zip(self.region, SEIR_list):
            part.set_SEIR_values(SEIR, t)
            self.S += SEIR[:,0]
            self.E1 += SEIR[:,1]
            self.E2 += SEIR[:,2]
            self.I += SEIR[:,3]
            self.Ia += SEIR[:,4]
            self.R += SEIR[:,5]

    def plot(self):
        plt.plot(self.t, self.S, label="S(t)")
        plt.plot(self.t, self.I, label="I(t)")
        plt.plot(self.t, self.Ia, label="Ia(t)")
        plt.plot(self.t, self.R, label="R(t)")
        plt.xlabel("Time(days)")
        plt.ylabel("Population")
        name = self.area_name
        plt.title(name)

if __name__ == "__main__":
    innlandet = RegionInteraction("Innlandet",S_0=371385, E2_0=0, lat=60.7945, long=11.0680)
    oslo = RegionInteraction("Oslo",S_0=693494,E2_0=100, lat=59.9, long=10.8)
    print(oslo.distance(innlandet))

    problem = ProblemInteraction([oslo,innlandet],"Norway_east", beta=0.5)
    print(problem.get_population())
    problem.set_initial_condition()
    print(problem.initial_condition) #non-nested list of length 12
    u = problem.initial_condition
    print(problem(u,0)) #list of length 12. Check that values make sense

    #when lines above work, add this code to solve a test problem:
    solver = SolverSEIR(problem,T=100,dt=1.0)
    solver.solve()
    problem.plot()
    plt.legend()
    plt.show()

"""
Terminal> python SEIR_interaction.py
1.0100809386285283
1064979
[693494, 0, 100, 0, 0, 0, 371385, 0, 0, 0, 0, 0]
[-62.490904683238746, 62.490904683238746, -50.0, 50.0, 0.0, 0.0, -12.1878323242723, 12.1878323242723, 0.0, 0.0, 0.0, 0.0]
"""
