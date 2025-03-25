import os
import matplotlib.pyplot as plt
import netket as nk
import jax
import numpy as np
from netket.operator.spin import sigmax, sigmaz 
from scipy.sparse.linalg import eigsh  # To calculate exact ground state energy
nk.config.netket_spin_ordering_warning = False

import jax.numpy as jnp

import flax.linen as nn
os.environ["JAX_PLATFORM_NAME"] = "cpu"

def confidence_interval(mean, stddev, n_samples):
    ci = 1.96 * (stddev / np.sqrt(n_samples))
    return mean - ci, mean + ci

class FFN(nn.Module):
    alpha: int = 1
    
    @nn.compact
    def __call__(self, x):
        dense = nn.Dense(features=self.alpha * x.shape[-1])
        y = dense(x)
        y = nn.relu(y)
        return jnp.sum(y, axis=-1)

N = 20
hi = nk.hilbert.Spin(s=1/2, N=N)
energies_mean = []
energies_lower_ci = []
energies_upper_ci = []
magnetizations_mean = []
magnetizations_lower_ci = []
magnetizations_upper_ci = []

Gammas = np.linspace(0, 2, 20, endpoint=True)

for Gamma in Gammas:
    exact_energy=[]
    Gamma = round(Gamma, 2)
    print(Gamma)
    H = sum([-Gamma * sigmax(hi, i) for i in range(N)])
    V = -1
    H += sum([V * sigmaz(hi, i) * sigmaz(hi, (i + 1) % N) for i in range(N)])

    sampler = nk.sampler.MetropolisLocal(hi)
    model = FFN(alpha=1)
    vstate = nk.vqs.MCState(sampler, model, n_samples=1008, seed=0)

    optimizer = nk.optimizer.Sgd(learning_rate=0.1)

    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.1))
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=300, out=log)

    # # Store energy per site over iterations for plotting
    iterations = log.data["Energy"].iters
    energies = log.data["Energy"].Mean / N 
    # errors = log.data["Energy"].Sigma

    # Calculate the exact ground state energy using sparse matrix diagonalization
    evals = nk.exact.lanczos_ed(H, compute_eigenvectors=False)
    exact = evals[0]
    for i in range(300):
        exact_energy.append(exact/N)
    print(exact)
    
    
    

    # Save data for this Gamma in a separate .npz file
    filename = f"Ising_energyvs_iter/energy_vs_iterations_Gamma{Gamma}.npz"
    np.savez(filename, 
             exact_energy=exact_energy,
             iterations=iterations,
             energies=energies)


    # mag_op = nk.operator.spin.sigmaz(hi, 0)
    # for i in range(1, N):
    #     mag_op += nk.operator.spin.sigmaz(hi, i)
    # magenergy = vstate.expect(mag_op)
    # mag_per_site = abs(magenergy.mean) / N
    # mag_stddev_per_site = magenergy.error_of_mean / N
    
    # mag_lower, mag_upper = confidence_interval(mag_per_site, mag_stddev_per_site, vstate.n_samples)
    
    # magnetizations_mean.append(mag_per_site)
    # magnetizations_lower_ci.append(mag_lower)
    # magnetizations_upper_ci.append(mag_upper)

# # Plotting Energy per site with 95% confidence interval
# plt.fill_between(Gammas, energies_lower_ci, energies_upper_ci, alpha=0.3, color="lightblue", label="95% CI")
# plt.plot(Gammas, energies_mean, color="blue", label="Mean Energy per Site")
# plt.xlabel("Transverse Field (Gamma)")
# plt.ylabel("Energy per Site")
# plt.title("Energy per Site vs Transverse Field")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plotting Magnetic Energy per site with 95% confidence interval
# plt.fill_between(Gammas, magnetizations_lower_ci, magnetizations_upper_ci, alpha=0.3, color="lightblue", label="95% CI")
# plt.plot(Gammas, magnetizations_mean, color="blue", label="Mean Magnetic Energy per Site")
# plt.xlabel("Transverse Field (Gamma)")
# plt.ylabel("Magnetic Energy per Site")
# plt.title("Magnetic Energy per Site vs Transverse Field")
# plt.legend()
# plt.grid(True)
# plt.show()
