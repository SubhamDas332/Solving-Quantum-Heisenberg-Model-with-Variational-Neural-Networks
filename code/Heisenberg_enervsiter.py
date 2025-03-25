import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import netket as nk
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import flax.linen as nn
import jax.numpy as jnp
import jax
from scipy.linalg import kron
nk.config.netket_spin_ordering_warning = False
from netket.operator.spin import sigmax,sigmaz,sigmay 

N = 20
hi = nk.hilbert.Spin(s=1/2, N=N)
def transverse_field_heisenberg_hamiltonian(N, J, h):


    # Initialize the Hamiltonian to zero
    # H = nk.operator.LocalOperator(hi)

    H = J * (sigmax(hi, 0) * sigmax(hi, 1) +
            sigmay(hi, 0) * sigmay(hi, 1) +
            sigmaz(hi, 0) * sigmaz(hi, 1))
    for i in range(1,N - 1):
        H += J * (sigmax(hi, i) * sigmax(hi, i+1) +
                  sigmay(hi, i) * sigmay(hi, i+1) +
                  sigmaz(hi, i) * sigmaz(hi, i+1))

    for i in range(N):
        H += -h * sigmaz(hi, i)

    return H

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*x.shape[-1], param_dtype=np.complex128, kernel_init=nn.initializers.normal(stddev=0.1), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nk.nn.activation.log_cosh(x)
        x = nn.Dense(features=x.shape[-1], param_dtype=np.complex128, kernel_init=nn.initializers.normal(stddev=0.1), bias_init=nn.initializers.normal(stddev=0.1))(x)
        x = nk.nn.activation.log_cosh(x)
        return jax.numpy.sum(x, axis=-1)



J = 1.0  # Coupling constant
h_s = np.linspace(0, 2, 20, endpoint=True)
for h in h_s:
    
    exact_energy= []
    H = transverse_field_heisenberg_hamiltonian(N, J, h)
    
    sampler = nk.sampler.MetropolisLocal(hi)
    ffnn = Model()
    sa =  nk.sampler.MetropolisLocal(hilbert=hi)
    op = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(diag_shift=0.1,holomorphic=True)
    vs = nk.vqs.MCState(sa, ffnn, n_samples=2048) 
    gs = nk.VMC(hamiltonian=H,
                optimizer=op,
                preconditioner=sr,
                variational_state=vs)
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=300, out=log)
    
    iterations = log.data["Energy"].iters
    energies = log.data["Energy"].Mean / N 
    
    evals = nk.exact.lanczos_ed(H, compute_eigenvectors=False)
    exact = evals[0]
    for i in range(300):
        exact_energy.append(exact/N)
    print(exact)
    
    filename = f"code\Heisenberg_energyvs_iter/energy_vs_iterations_h1_{h}.npz"
    np.savez(filename, 
             exact_energy=exact_energy,
             iterations=iterations,
             energies=energies)
    


    # # Plotting the energy vs iterations
    # plt.figure(figsize=(10, 6))
    # plt.plot(iterations, energies, label='Energy (per site)', color='blue', linestyle='-', marker='o')
    # plt.axhline(y=exact / N, color='red', linestyle='--', label='Exact Energy (per site)')

    # plt.xlabel('Iterations', fontsize=14)
    # plt.ylabel('Energy (per site)', fontsize=14)
    # plt.title('Energy vs Iterations', fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid(True)
    # plt.show()
    
    
    

    