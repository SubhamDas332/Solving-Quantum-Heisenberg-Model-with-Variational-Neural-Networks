import os
# os.environ["JAX_PLATFORM_NAME"] = "GPU"
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
from scipy.sparse import csr_array


def calculate_magnetization(vs, hi, direction="z"):
    if direction == "z":
        op = sum(sigmaz(hi, i) for i in range(hi.size))
    elif direction == "x":
        op = sum(sigmax(hi, i) for i in range(hi.size))
    else:
        raise ValueError("Invalid direction: choose 'x' or 'z'")
    
    if isinstance(vs, nk.vqs.MCState):  # Variational state
        return vs.expect(op).mean / hi.size
    elif isinstance(vs, np.ndarray):  # Exact eigenstate
        op=op.to_sparse()
        vs= csr_array(vs)
        # Use sparse matrix multiplication for exact eigenstates
        expectation_value = vs.conj().transpose().dot(op.dot(vs))
        return expectation_value / hi.size


def calculate_susceptibility(vs, hi, direction="z"):
    if direction == "z":
        op = sum(sigmaz(hi, i) for i in range(hi.size))
    elif direction == "x":
        op = sum(sigmax(hi, i) for i in range(hi.size))
    else:
        raise ValueError("Invalid direction: choose 'x' or 'z'")
    
    if isinstance(vs, nk.vqs.MCState):  # Variational state
        op_squared = op @ op
        expectation_value = vs.expect(op).mean
        expectation_value_squared = vs.expect(op_squared).mean
    elif isinstance(vs, np.ndarray):  # Exact eigenstate
        op_squared = op @ op
        op=op.to_sparse()
        op_squared=op_squared.to_sparse()
        vs= csr_array(vs)
        expectation_value =  vs.conj().transpose().dot(op.dot(vs))
        expectation_value_squared =  vs.conj().transpose().dot(op_squared.dot(vs))
    
    susceptibility = (expectation_value_squared - expectation_value**2) / hi.size
   
    return susceptibility

def transverse_field_ising_hamiltonian(N, J, h):
    H = sum([-h*sigmax(hi,i) for i in range(N)])
    J=-1
    H += sum([J*sigmaz(hi,i)*sigmaz(hi,(i+1)%N) for i in range(N)])
    return H

class Model(nn.Module):
    alpha: int = 1
    @nn.compact
    def __call__(self, x):
        dense = nn.Dense(features=self.alpha * x.shape[-1])
        y = dense(x)
        y = nn.relu(y)
        return jnp.sum(y, axis=-1)

def run_vmc_for_field_strengths(field_values, hi, J):
    energy_per_site = []
    energy_per_site_ci_lower = []
    energy_per_site_ci_upper = []
    magnetization_z = []
    magnetization_x = []
    susceptibility_z = []
    susceptibility_x = []
    exact_gs_energiespersite = []
    Transverse = []
    exact_magnetisation_z=[]
    exact_susceptibility_z=[]

    for h in field_values:
        H = transverse_field_ising_hamiltonian(hi.size, J, h)
        evals,evec = nk.exact.lanczos_ed(H,compute_eigenvectors=True,k=1)
        # evec = evec / np.linalg.norm(evec)
        exact_gs_energypersite = evals[0] / hi.size
        exact_magnetisation_per_site=calculate_magnetization(evec,hi)
        exact_susceptibility=calculate_susceptibility(evec,hi)
        
        ffnn = Model()
        sa = nk.sampler.MetropolisLocal(hilbert=hi)
        op = nk.optimizer.Sgd(learning_rate=0.05)
        sr = nk.optimizer.SR(diag_shift=0.1)
        vs = nk.vqs.MCState(sa, ffnn, n_samples=2048)
        gs = nk.VMC(hamiltonian=H, optimizer=op, preconditioner=sr, variational_state=vs)
        gs.run(n_iter=300)

        # Calculate energy and its confidence interval
        energy = vs.expect(H).mean / hi.size
        energy_stddev = vs.expect(H).variance ** 0.5 / hi.size
        confidence_interval = 1.96 * (energy_stddev / np.sqrt(vs.n_samples))

        # Store lower and upper bounds of the 95% confidence interval
        energy_ci_lower = energy - confidence_interval
        energy_ci_upper = energy + confidence_interval

        # Magnetization and susceptibility
        mz = calculate_magnetization(vs, hi, direction="z")
        mx = calculate_magnetization(vs, hi, direction="x")
        chi_z = calculate_susceptibility(vs, hi, direction="z")
        chi_x = calculate_susceptibility(vs, hi, direction="x")

        # Store results
        exact_gs_energiespersite.append(exact_gs_energypersite)
        energy_per_site.append(energy)
        energy_per_site_ci_lower.append(energy_ci_lower)
        energy_per_site_ci_upper.append(energy_ci_upper)
        magnetization_z.append(mz)
        magnetization_x.append(mx)
        susceptibility_z.append(chi_z)
        susceptibility_x.append(chi_x)
        Transverse.append(h)
        exact_magnetisation_z.append(exact_magnetisation_per_site)
        exact_susceptibility_z.append(exact_susceptibility)

    return Transverse, energy_per_site, energy_per_site_ci_lower, energy_per_site_ci_upper, magnetization_z, magnetization_x, susceptibility_z, susceptibility_x, exact_gs_energiespersite ,exact_magnetisation_z,exact_susceptibility_z

field_values = np.linspace(0, 2, 2)
N = 10
hi = nk.hilbert.Spin(s=1/2, N=N)
J = 1.0  # Coupling constant

for i in range(1, 2):
    Transverse, energy_per_site, energy_per_site_ci_lower, energy_per_site_ci_upper, magnetization_z, magnetization_x, susceptibility_z, susceptibility_x, exact_gs_energiespersite ,exact_magnetisation_z,exact_susceptibility_z = run_vmc_for_field_strengths(field_values, hi, J)
    print('Energy per site:', energy_per_site, 'Exact energies per site:', exact_gs_energiespersite)
    np.savez(f"code\\Ising_data_plot_latest\\ising_run_{i}.npz",
             Transverse=Transverse, 
             energy_per_site=energy_per_site,
             energy_per_site_ci_lower=energy_per_site_ci_lower,
             energy_per_site_ci_upper=energy_per_site_ci_upper,
             magnetization_z=magnetization_z,
             magnetization_x=magnetization_x,
             susceptibility_z=susceptibility_z,
             susceptibility_x=susceptibility_x,
             exact_gs_energiespersite=exact_gs_energiespersite,
             exact_magnetisation_z=exact_magnetisation_z,
             exact_susceptibility_z=exact_susceptibility_z)
    print(f"Data for run {i} finished and saved with confidence intervals.")
