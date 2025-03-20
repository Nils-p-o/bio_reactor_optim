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

Ks = 0.17  # g/l gotten from gemini (AI) as an estimate (0.034 - 0.342)


def plug_flow(step, end_time, X0, S0, P0, method="euler"):
    X1 = X0
    S1 = S0
    P1 = P0
    F = V / end_time

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
    productivity = P1 * 1 / end_time
    if S1 < 0:
        productivity = 0
        substrate_utility = 0
    # print("X1 shape:", X1.shape)
    # print("S1 shape:", S1.shape)
    # print("P1 shape:", P1.shape)
    # print("substrate_utility shape:", substrate_utility.shape)
    # print("productivity shape:", productivity.shape)

    return np.array([X1, S1, P1, substrate_utility, productivity])


def plug_flow_vectorized(step, end_time, X0, S0, method="euler"):
    results = np.zeros(
        (len(X0), len(S0), len(end_time), 6)
    )  # X, S, P, substrate_utility, productivity, end_time
    S0 = np.tile(S0.reshape(1, len(S0), 1), (len(X0), 1, len(end_time)))
    X0 = np.tile(X0.reshape(len(X0), 1, 1), (1, len(S0), len(end_time)))
    end_time = np.tile(end_time.reshape(1, 1, len(end_time)), (len(X0), len(S0), 1))

    results[:, :, :, 5] = end_time.copy()
    results[:, :, :, 1] = S0.copy()
    results[:, :, :, 0] = X0.copy()

    def derivatives(X, S, P):
        mu = mu_max * (1 - X / X_max) * S / (Ks + S)
        q_s = delta * mu + ni
        q_p = alpha * mu + beta
        dXdt = mu * X
        dSdt = -q_s * X
        dPdt = q_p * X
        return dXdt, dSdt, dPdt

    max_steps = round(np.max(end_time) / step)

    # euler method
    if method == "euler":
        for i in range(max_steps):
            dXdt, dSdt, dPdt = derivatives(
                results[:, :, :, 0], results[:, :, :, 1], results[:, :, :, 2]
            )
            if i % 50 == 0:
                print(f"{i}/{max_steps}")
                print("mean:", np.mean(dXdt), np.mean(dSdt), np.mean(dPdt))
            results[:, :, :, 0] = np.where(
                results[:, :, :, 5] > i * step,
                results[:, :, :, 0] + step * dXdt,
                results[:, :, :, 0],
            )
            results[:, :, :, 1] = np.where(
                results[:, :, :, 5] > i * step,
                results[:, :, :, 1] + step * dSdt,
                results[:, :, :, 1],
            )
            results[:, :, :, 2] = np.where(
                results[:, :, :, 5] > i * step,
                results[:, :, :, 2] + step * dPdt,
                results[:, :, :, 2],
            )

    # runge-kutta 4th order
    elif method == "rk4":
        for i in range(max_steps):
            if i % 50 == 0:
                print(f"{i}/{max_steps}")
            k1_X, k1_S, k1_P = derivatives(
                results[:, :, :, 0], results[:, :, :, 1], results[:, :, :, 2]
            )
            k2_X, k2_S, k2_P = derivatives(
                results[:, :, :, 0] + 0.5 * step * k1_X,
                results[:, :, :, 1] + 0.5 * step * k1_S,
                results[:, :, :, 2] + 0.5 * step * k1_P,
            )
            k3_X, k3_S, k3_P = derivatives(
                results[:, :, :, 0] + 0.5 * step * k2_X,
                results[:, :, :, 1] + 0.5 * step * k2_S,
                results[:, :, :, 2] + 0.5 * step * k2_P,
            )
            k4_X, k4_S, k4_P = derivatives(
                results[:, :, :, 0] + step * k3_X,
                results[:, :, :, 1] + step * k3_S,
                results[:, :, :, 2] + step * k3_P,
            )
            results[:, :, :, 0] = np.where(
                results[:, :, :, 5] > i * step,
                results[:, :, :, 0] + (step / 6) * (k1_X + 2 * k2_X + 2 * k3_X + k4_X),
                results[:, :, :, 0],
            )
            results[:, :, :, 1] = np.where(
                results[:, :, :, 5] > i * step,
                results[:, :, :, 1] + (step / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S),
                results[:, :, :, 1],
            )
            results[:, :, :, 2] = np.where(
                results[:, :, :, 5] > i * step,
                results[:, :, :, 2] + (step / 6) * (k1_P + 2 * k2_P + 2 * k3_P + k4_P),
                results[:, :, :, 2],
            )

    results[:, :, :, 3] = np.where(
        results[:, :, :, 1] > 0, (S0 - results[:, :, :, 1]) / S0, 0
    )
    results[:, :, :, 4] = np.where(
        results[:, :, :, 1] > 0, results[:, :, :, 2] / results[:, :, :, 5], 0
    )

    return results


sim_result = np.zeros((100, 100, 200, 6))

x_values = np.linspace(start=1, stop=X_max, num=100)
s_values = np.linspace(start=0.1, stop=20, num=100)
time_values = np.linspace(start=0.1, stop=100, num=200)

sim_result = plug_flow_vectorized(
    step=0.01, end_time=time_values, X0=x_values, S0=s_values, method="rk4"
)

# old method (only productivity)

opt_productivity = np.max(sim_result[:, :, :, 4])
opt_productivity_index = np.where(sim_result[:, :, :, 4] == opt_productivity)
opt_productivity_X = x_values[opt_productivity_index[0][0]]
opt_productivity_S = s_values[opt_productivity_index[1][0]]
opt_productivity_T = time_values[opt_productivity_index[2][0]]
# print(opt_productivity_index)
print(f"Optimal productivity: {opt_productivity}")
print(f"Optimal X: {opt_productivity_X}")
print(f"Optimal S: {opt_productivity_S}")
print(f"Optimal T: {opt_productivity_T}")

print(
    sim_result[
        opt_productivity_index[0][0],
        opt_productivity_index[1][0],
        opt_productivity_index[2][0],
        :,
    ]
)


# figure, axis = plt.subplots(1, 3)
# # plt.imshow(sim_result[:,:,3])
# # plt.show()
# # axis[0].imshow(np.average(sim_result[:, :, :, 4], axis=2))
# axis[0].imshow(sim_result[:, :, opt_productivity_index[2][0], 4])
# axis[0].set_title("productivity (x,s)")

# # axis[1].imshow(np.average(sim_result[:, :, :, 4], axis=1))
# axis[1].imshow(sim_result[:, opt_productivity_index[1][0], :, 4])
# axis[1].set_title("productivity (x,t)")

# # axis[2].imshow(np.average(sim_result[:, :, :, 4], axis=0))
# axis[2].imshow(sim_result[opt_productivity_index[0][0], :, :, 4])
# axis[2].set_title("productivity (s,t)")

# plt.show()

# productivity + substrate utilization

opt_productivity = np.max(sim_result[:, :, :, 4] * sim_result[:, :, :, 3])
opt_productivity_index = np.where(
    sim_result[:, :, :, 4] * sim_result[:, :, :, 3] == opt_productivity
)
opt_productivity_X = x_values[opt_productivity_index[0][0]]
opt_productivity_S = s_values[opt_productivity_index[1][0]]
opt_productivity_T = time_values[opt_productivity_index[2][0]]

print(
    f"Optimal productivity: {sim_result[opt_productivity_index[0][0], opt_productivity_index[1][0], opt_productivity_index[2][0], 4]}"
)
print(
    f"Optimal substrate utilization: {sim_result[opt_productivity_index[0][0], opt_productivity_index[1][0], opt_productivity_index[2][0], 3]}"
)
print(f"Optimal X: {opt_productivity_X}")
print(f"Optimal S: {opt_productivity_S}")
print(f"Optimal T: {opt_productivity_T}")

print(
    sim_result[
        opt_productivity_index[0][0],
        opt_productivity_index[1][0],
        opt_productivity_index[2][0],
        :,
    ]
)

figure, axis = plt.subplots(3, 3)
# plt.imshow(sim_result[:,:,3])
# plt.show()
# axis[0].imshow(np.average(sim_result[:, :, :, 4], axis=2))
axis[0, 0].imshow(sim_result[:, :, opt_productivity_index[2][0], 4])
axis[0, 0].set_title("productivity (x,s)")

# axis[1].imshow(np.average(sim_result[:, :, :, 4], axis=1))
axis[0, 1].imshow(sim_result[:, opt_productivity_index[1][0], :, 4])
axis[0, 1].set_title("productivity (x,t)")

# axis[2].imshow(np.average(sim_result[:, :, :, 4], axis=0))
axis[0, 2].imshow(sim_result[opt_productivity_index[0][0], :, :, 4])
axis[0, 2].set_title("productivity (s,t)")


axis[1, 0].imshow(sim_result[:, :, opt_productivity_index[2][0], 3])
axis[1, 0].set_title("substrate utilization (x,s)")

axis[1, 1].imshow(sim_result[:, opt_productivity_index[1][0], :, 3])
axis[1, 1].set_title("substrate utilization (x,t)")

axis[1, 2].imshow(sim_result[opt_productivity_index[0][0], :, :, 3])
axis[1, 2].set_title("substrate utilization (s,t)")

# Productivity * substrate utilization
axis[2, 0].imshow(
    sim_result[:, :, opt_productivity_index[2][0], 4]
    * sim_result[:, :, opt_productivity_index[2][0], 3]
)
axis[2, 0].set_title("productivity * substrate utilization (x,s)")

axis[2, 1].imshow(
    sim_result[:, opt_productivity_index[1][0], :, 4]
    * sim_result[:, opt_productivity_index[1][0], :, 3]
)
axis[2, 1].set_title("productivity * substrate utilization (x,t)")

axis[2, 2].imshow(
    sim_result[opt_productivity_index[0][0], :, :, 4]
    * sim_result[opt_productivity_index[0][0], :, :, 3]
)
axis[2, 2].set_title("productivity * substrate utilization (s,t)")

plt.show()
