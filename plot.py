import numpy as np
import matplotlib.pyplot as plt

X_max = 9.67  # g/l
mu_max = 0.2653  # 1/h
Y_xs = 1.934  # g/g
delta = 0.5155  # g/g
ni = 0.0215  # g/gh
alpha = 0.4313  # g/g
beta = 0.0179  # g/gh
V = 100  # m^3

Ks = 0.17  # g/l gotten from gemini as an estimate (0.034 - 0.342)


def plug_flow(step, end_time, method="euler", X0=0, S0=10, P0=0):
    X1 = X0
    S1 = S0
    P1 = P0
    F = V / end_time
    data = np.zeros((round(end_time / step), 2))

    def derivatives(X, S, P):
        mu = mu_max * (1 - X / X_max) * S / (Ks + S)
        q_s = delta * mu + ni
        q_p = alpha * mu + beta
        dXdt = mu * X
        dSdt = -q_s * X
        dPdt = q_p * X
        return dXdt, dSdt, dPdt

    # euler method
    if method == "euler":
        for i in range(round(end_time / step)):
            dXdt, dSdt, dPdt = derivatives(X1, S1, P1)
            X1 = X1 + step * dXdt
            S1 = S1 + step * dSdt
            P1 = P1 + step * dPdt
            substrate_utility = (S0 - S1) / S0
            productivity = P1 / (i * step + 1e-9)
            if i == 0:
                substrate_utility = 0
                productivity = 0
            data[i] = [(i+1)*step, X1]#, S1, P1, substrate_utility*10, productivity*10]

    # runge-kutta 4th order
    elif method == "rk4":
        for i in range(round(end_time / step)):
            k1_X, k1_S, k1_P = derivatives(X1, S1, P1)
            k2_X, k2_S, k2_P = derivatives(
                X1 + 0.5 * step * k1_X, S1 + 0.5 * step * k1_S, P1 + 0.5 * step * k1_P
            )
            k3_X, k3_S, k3_P = derivatives(
                X1 + 0.5 * step * k2_X, S1 + 0.5 * step * k2_S, P1 + 0.5 * step * k2_P
            )
            k4_X, k4_S, k4_P = derivatives(
                X1 + step * k3_X, S1 + step * k3_S, P1 + step * k3_P
            )

            X1 = X1 + (step / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
            S1 = S1 + (step / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
            P1 = P1 + (step / 6) * (k1_P + 2 * k2_P + 2 * k3_P + k4_P)

            substrate_utility = (S0 - S1) / S0
            productivity = P1 / (i * step + 1e-9)
            if i == 0:
                substrate_utility = 0
                productivity = 0
            data[i] = [(i+1)*step, X1]#, S1, P1, substrate_utility*10, productivity*10]



    return X1, S1, P1, substrate_utility, productivity, data
    # runge-kutta 4th order


sim_result = np.zeros((50, 10, 30, 5))

# x_values = np.linspace(start=0.01, stop=X_max)
# s_values = np.linspace(start=1, stop=10, num=10)
# time_values = np.linspace(start=1, stop=30, num=10)
# x0 = 3.62
# s0 = 3.32
# t = 8.867
x0 = 0.5
s0 = 5
t = 10

x_graph_result_euler = [0,0,0,0]
x_graph_result_euler[0] = plug_flow(step=1., end_time=t, X0=x0, S0=s0)[5]
x_graph_result_euler[1] = plug_flow(step=0.1, end_time=t, X0=x0, S0=s0)[5]
x_graph_result_euler[2] = plug_flow(step=0.01, end_time=t, X0=x0, S0=s0)[5]
x_graph_result_euler[3] = plug_flow(step=0.001, end_time=t, X0=x0, S0=s0)[5]

x_graph_result_rk4 = [0,0,0,0]
x_graph_result_rk4[0] = plug_flow(step=1., end_time=t, X0=x0, S0=s0, method="rk4")[5]
x_graph_result_rk4[1] = plug_flow(step=0.1, end_time=t, X0=x0, S0=s0, method="rk4")[5]
x_graph_result_rk4[2] = plug_flow(step=0.01, end_time=t, X0=x0, S0=s0, method="rk4")[5]
x_graph_result_rk4[3] = plug_flow(step=0.001, end_time=t, X0=x0, S0=s0, method="rk4")[5]

plt.plot(x_graph_result_euler[0][:, 0], x_graph_result_euler[0][:, 1])
plt.plot(x_graph_result_euler[1][:, 0], x_graph_result_euler[1][:, 1])
plt.plot(x_graph_result_euler[2][:, 0], x_graph_result_euler[2][:, 1])
plt.plot(x_graph_result_euler[3][:, 0], x_graph_result_euler[3][:, 1])
plt.xlabel("time (h)")
plt.ylabel("X")
plt.legend(["Euler 1.0", "Euler 0.1", "Euler 0.01", "Euler 0.001"])
plt.savefig("bio_rector_optimization/plots/euler_method.png")
plt.show()


plt.plot(x_graph_result_rk4[0][:, 0], x_graph_result_rk4[0][:, 1])
plt.plot(x_graph_result_rk4[1][:, 0], x_graph_result_rk4[1][:, 1])
plt.plot(x_graph_result_rk4[2][:, 0], x_graph_result_rk4[2][:, 1])
plt.plot(x_graph_result_rk4[3][:, 0], x_graph_result_rk4[3][:, 1])
plt.xlabel("time (h)")
plt.ylabel("X")
plt.legend(["RK4 1.0", "RK4 0.1", "RK4 0.01", "RK4 0.001"])
plt.savefig("bio_rector_optimization/plots/rk4_method.png")
plt.show()

plt.plot(x_graph_result_euler[0][:, 0], x_graph_result_euler[0][:, 1])
plt.plot(x_graph_result_euler[3][:, 0], x_graph_result_euler[3][:, 1])
plt.plot(x_graph_result_rk4[0][:, 0], x_graph_result_rk4[0][:, 1])
plt.xlabel("time (h)")
plt.ylabel("X")
plt.legend(["Euler 1.0", "Euler 0.001", "RK4 1.0"])
plt.savefig("bio_rector_optimization/plots/euler_vs_rk4_method.png")
plt.show()

# sim_result[1, 0, 0, :] = plug_flow(step=1., end_time=t, X0=x0, S0=s0)

# sim_result[1, 0, 1, :] = plug_flow(step=0.1, end_time=t, X0=x0, S0=s0)

# sim_result[1, 0, 2, :] = plug_flow(step=0.01, end_time=t, X0=x0, S0=s0)

# sim_result[1, 0, 3, :] = plug_flow(step=0.001, end_time=t, X0=x0, S0=s0)

# sim_result[1, 2, 0, :] = plug_flow(step=1., end_time=t, X0=x0, S0=s0, method="rk4")

# sim_result[1, 2, 1, :] = plug_flow(step=0.1, end_time=t, X0=x0, S0=s0, method="rk4") 

# sim_result[1, 2, 2, :] = plug_flow(step=0.01, end_time=t, X0=x0, S0=s0, method="rk4")

# sim_result[1, 2, 3, :] = plug_flow(step=0.001, end_time=t, X0=x0, S0=s0, method="rk4")
