#Problem 1.2. Introduce classes in the SEIR model
from ODESolver import *
import matplotlib.pyplot as plt
import numpy as np

#a)
class Region:
    def __init__(self, name, S_0, E2_0):
        self.name = name
        self.S0 = S_0
        self.E1_0 = 0
        self.E2_0 = E2_0
        self.I0 = 0
        self.Ia_0 = 0
        self.R_0 = 0
        self.population = S_0 + E2_0
    def set_SEIR_values(self, u, t):
        self.S = u[:,0]
        self.E1 = u[:,1]
        self.E2 = u[:,2]
        self.I = u[:,3]
        self.Ia = u[:,4]
        self.R = u[:,5]
        self.t = t
    def plot(self):
        plt.plot(self.t, self.S, label="S(t)")
        plt.plot(self.t, self.I, label="I(t)")
        plt.plot(self.t, self.Ia, label="Ia(t)")
        plt.plot(self.t, self.R, label="R(t)")
        plt.xlabel("Time(days)")
        plt.ylabel("Population")
        name = self.name
        plt.title(name)

#b)
class ProblemSEIR:
    def __init__(self, region, beta, r_ia = 0.1, r_e2=1.25, lmbda_1=0.33, lmbda_2=0.5, p_a=0.4, mu=0.2):
        if isinstance(beta, (float, int)):
            self.beta = lambda t: beta
        elif callable(beta):
            self.beta = beta
        self.region = region
        self.r_ia = r_ia
        self.r_e2 = r_e2
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2
        self.p_a = p_a
        self.mu = mu
    def set_initial_condition(self):
        self.initial_condition = [self.region.S0, self.region.E1_0, self.region.E2_0, self.region.I0, self.region.Ia_0, self.region.R_0]
    def get_population(self):
        return self.region.population
    def solution(self, u, t):
        return self.region.set_SEIR_values(u,t)
    def __call__(self, u, t):
        self.S, self.E1, self.E2, self.I, self.Ia, self.R = u
        N = sum(u)
        dS = -self.beta(t)*self.S*self.I/N - self.r_ia*self.beta(t)*self.S*self.Ia/N - self.r_e2*self.beta(t)*self.S*self.E2/N
        dE1 = self.beta(t)*self.S*self.I/N + self.r_ia*self.beta(t)*self.S*self.Ia/N + self.r_e2*self.beta(t)*self.S*self.E2/N - self.lmbda_1*self.E1
        dE2 = self.lmbda_1*(1-self.p_a)*self.E1 - self.lmbda_2*self.E2
        dI = self.lmbda_2*self.E2 - self.mu*self.I
        dIa = self.lmbda_1*self.p_a*self.E1 - self.mu*self.Ia
        dR = self.mu*(self.I + self.Ia)
        return [dS, dE1, dE2, dI, dIa, dR]

#c
class SolverSEIR:
    def __init__(self,problem,T,dt):
        self.problem = problem
        self.T = T
        self.dt = dt
        self.total_population = problem.get_population()
    def solve(self, method=RungeKutta4):
        solver = method(self.problem)
        solver.set_initial_condition(self.problem.initial_condition)
        N = int(self.T/self.dt)+1
        t = np.linspace(0,self.T,N)
        u, t = solver.solve(t)
        self.problem.solution(u, t)

if __name__ == "__main__":

    nor = Region("Norway",S_0=5e6,E2_0=100)
    print(nor.name, nor.population)
    S_0, E1_0, E2_0 = nor.S0, nor.E1_0, nor.E2_0
    I_0, Ia_0, R_0 = nor.I0, nor.Ia_0, nor.R_0
    print(f"S_0 = {S_0}, E1_0 = {E1_0}, E2_0 = {E2_0}")
    print(f"I_0 = {I_0}, Ia_0 = {Ia_0}, R_0 = {R_0}")
    u = np.zeros((2,6))
    u[0,:] = [S_0, E1_0, E2_0, I_0, Ia_0, R_0]
    nor.set_SEIR_values(u,0)
    print(nor.S, nor.E1, nor.E2, nor.I, nor.Ia, nor.R)

    problem = ProblemSEIR(nor,beta=0.5)
    problem.set_initial_condition()
    print(problem.initial_condition)
    print(problem.get_population())
    print(problem([1,1,1,1,1,1],0))

    solver = SolverSEIR(problem,T=100,dt=1.0)
    solver.solve()
    nor.plot()
    plt.legend()
    plt.show()


"""
Terminal> python SEIR.py
Norway 5000100.0
S_0 = 5000000.0, E1_0 = 0, E2_0 = 100
I_0 = 0, Ia_0 = 0, R_0 = 0
[5000000.       0.] [0. 0.] [100.   0.] [0. 0.] [0. 0.] [0. 0.]
[5000000.0, 0, 100, 0, 0, 0]
5000100.0
[-0.19583333333333333, -0.13416666666666668, -0.302, 0.3, -0.068, 0.4]
"""
