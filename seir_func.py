#Problem 1.1. The SEIR model as a function
from math import e
from ODESolver import *
import numpy as np
import matplotlib.pyplot as plt

#a)
def SEIR(u,t):
    beta = 0.5; r_ia = 0.1; r_e2=1.25;
    lmbda_1=0.33; lmbda_2=0.5; p_a=0.4; mu=0.2;

    S, E1, E2, I, Ia, R = u
    N = sum(u)
    dS  = -beta*S*I/N - r_ia*beta*S*Ia/N - r_e2*beta*S*E2/N
    dE1 = beta*S*I/N + r_ia*beta*S*Ia/N + r_e2*beta*S*E2/N - lmbda_1*E1
    dE2 = lmbda_1*(1-p_a)*E1 - lmbda_2*E2
    dI  = lmbda_2*E2 - mu*I
    dIa = lmbda_1*p_a*E1 - mu*Ia
    dR  = mu*(I + Ia)
    return [dS, dE1, dE2, dI, dIa, dR]

def test_SEIR():
    t = 0
    u = [1, 1, 1, 1, 1, 1]
    output = [-0.19583333333333333, -0.13416666666666668, -0.302, 0.3, -0.068, 0.4]
    tol = 1*e**-10
    computed = SEIR(u,t)
    for val, out in zip (computed, output):
        assert abs(val-out) < tol, "Test failed."
a = test_SEIR()

#b)
def solve_SEIR(T, dt, S_0, E2_0):
    initial = [S_0, 0, E2_0, 0, 0, 0]
    s = RungeKutta4(SEIR)
    s.set_initial_condition(initial)
    a = int(int(T)/int(dt))+1
    time_points = np.linspace(0, T, a)
    u, t = s.solve(time_points)
    return u, t

#c)
def plot_SEIR(u, t):
    S = u[:,0]
    I = u[:,3]
    Ia = u[:,4]
    R = u[:,5]
    plt.plot(t,S,label="S(t)")
    plt.plot(t,I,label="I(t)")
    plt.plot(t,Ia,label="Ia(t)")
    plt.plot(t,R,label="R(t)")
    plt.legend()
    plt.show()
    return

u, t = solve_SEIR(100, 1.0, 5*e**6, 100)
plot_SEIR(u, t)

"""
Terminal> python seir_func.py
"""
