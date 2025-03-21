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


def FSCR(X1, S0):
    mu = mu_max * (1 - X1 / X_max) * S0 / (Ks + S0) # should be fixed (S1 not S0)
    q_s = delta * mu + ni
    q_p = alpha * mu + beta
    S1 = S0 - q_s * X1
    P1 = q_p * X1

    substrate_utility = (S0 - S1)/S0
    productivity = P1 * mu
    if S1 < 0:
        substrate_utility = 0
        productivity = 0
    return S1, P1, substrate_utility, productivity


sim_result = np.zeros((50, 20, 4))

x_values = np.linspace(start=0, stop=X_max)
s_values = np.linspace(start=0.1, stop=10, num=20)

for i in range(len(x_values)):
    for j in range(len(s_values)):
        sim_result[i, j, :] = FSCR(x_values[i], s_values[j])

# figure, axis = plt.subplots(1, 2)
# plt.imshow(sim_result[:,:,3])
# plt.show()
# axis[0].imshow(sim_result[:, :, 3])
# axis[0].set_title("productivity")

# axis[1].imshow(sim_result[:, :, 2])
# axis[1].set_title("substrate utility")

# plt.show()

plt.imshow(sim_result[:, :, 3])
plt.colorbar()
plt.xlabel("S")
plt.ylabel("X")
plt.savefig("plots/FSCR_optimum/productivity.png")
plt.show()

plt.imshow(sim_result[:, :, 3] * sim_result[:, :, 2])
plt.colorbar()
plt.xlabel("S")
plt.ylabel("X")
plt.savefig("plots/FSCR_optimum/productivity_utility.png")
plt.show()

opt_productivity = np.max(sim_result[:, :, 3])
opt_productivity_index = np.where(sim_result[:, :, 3] == opt_productivity)
opt_productivity_X = x_values[opt_productivity_index[0][0]]
opt_productivity_S = s_values[opt_productivity_index[1][0]]
print(f"Optimal productivity: {opt_productivity}")
print(f"Optimal X: {opt_productivity_X}")
print(f"Optimal S: {opt_productivity_S}")

plt.plot(s_values, sim_result[opt_productivity_index[0][0], :, 3])
plt.xlabel("S")
plt.ylabel("productivity")
plt.savefig("plots/FSCR_optimum/productivity_S.png")
plt.show()

plt.plot(x_values, sim_result[:, opt_productivity_index[1][0], 3])
plt.xlabel("X")
plt.ylabel("productivity")
plt.savefig("plots/FSCR_optimum/productivity_X.png")
plt.show()

opt_productivity = np.max(sim_result[:, :, 2]*sim_result[:, :, 3])
opt_productivity_index = np.where(sim_result[:, :, 2]*sim_result[:, :, 3] == opt_productivity)
opt_productivity_X = x_values[opt_productivity_index[0][0]]
opt_productivity_S = s_values[opt_productivity_index[1][0]]
print(f"Optimal productivity: {sim_result[opt_productivity_index[0][0], opt_productivity_index[1][0], 3]}")
print(f"Optimal substrate utility: {sim_result[opt_productivity_index[0][0], opt_productivity_index[1][0], 2]}")
print(f"Optimal X: {opt_productivity_X}")
print(f"Optimal S: {opt_productivity_S}")

plt.plot(s_values, sim_result[opt_productivity_index[0][0], :, 2]*sim_result[opt_productivity_index[0][0], :, 3])
plt.xlabel("S")
plt.ylabel("productivity*substrate utility")
plt.savefig("plots/FSCR_optimum/productivity_utility_S.png")
plt.show()

plt.plot(x_values, sim_result[:, opt_productivity_index[1][0], 2]*sim_result[:, opt_productivity_index[1][0], 3])
plt.xlabel("X")
plt.ylabel("productivity*substrate utility")
plt.savefig("plots/FSCR_optimum/productivity_utility_X.png")
plt.show()