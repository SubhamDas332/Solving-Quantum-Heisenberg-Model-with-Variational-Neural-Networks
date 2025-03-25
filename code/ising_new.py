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



def calculate_magnetization(vs, hi, direction="z"):

    if direction == "z":
        op = sum(sigmaz(hi, i) for i in range(hi.size))
    elif direction == "x":
        op = sum(sigmax(hi, i) for i in range(hi.size))
    else:
        raise ValueError("Invalid direction: choose 'x' or 'z'")

    return vs.expect(op).mean / hi.size  # Magnetization per site

def calculate_susceptibility(vs, hi, direction="z"):

    if direction == "z":
        op = sum(sigmaz(hi, i) for i in range(hi.size))
    elif direction == "x":
        op = sum(sigmax(hi, i) for i in range(hi.size))
    else:
        raise ValueError("Invalid direction: choose 'x' or 'z'")

    # magnetization operator squared
    op_squared = op @ op
    expectation_value = vs.expect(op).mean
    expectation_value_squared = vs.expect(op_squared).mean
    susceptibility = (expectation_value_squared - expectation_value**2) / hi.size
    return susceptibility


def transverse_field_ising_hamiltonian(N, J, h):


    # Initialize the Hamiltonian to zero
    # H = nk.operator.LocalOperator(hi)

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
    magnetization_z = []
    magnetization_x = []
    susceptibility_z = []
    susceptibility_x = []
    exact_gs_energiespersite=[]
    Transverse=[]
    exact_magnetisation=[]

    for h in field_values:

        H = transverse_field_ising_hamiltonian(hi.size, J, h)
        evals,evec = nk.exact.lanczos_ed(H, compute_eigenvectors=True,k=1)
        exact_gs_energypersite = evals[0]/hi.size
        exact_magnetisation_per_site=calculate_magnetization(evec,hi)# exact_gs_energy = -39.14752260706246
                                            # exact_gs_energy = -34.72989333759587  #N=20
                                            # exact_gs_energy =  -17.032140829131475
        # print('exact ground state',exact_gs_energypersite)

        ffnn = Model()
        sa =  nk.sampler.MetropolisLocal(hilbert=hi)
        op = nk.optimizer.Sgd(learning_rate=0.05) #,nesterov=True
        sr = nk.optimizer.SR(diag_shift=0.1)
        vs = nk.vqs.MCState(sa, ffnn, n_samples=2048) #seed=ranint, sampler_seed=ranint
        gs = nk.VMC(hamiltonian=H,
                    optimizer=op,
                    preconditioner=sr,
                    variational_state=vs)
        gs.run(n_iter=300)
        #290037 for n=6
        #504081 for n=4
        # Calculate energy, magnetization, and susceptibility
        energy = vs.expect(H).mean/ hi.size
        mz = calculate_magnetization(vs,  hi, direction="z")
        mx = calculate_magnetization(vs,  hi, direction="x")
        chi_z = calculate_susceptibility(vs,  hi, direction="z")
        chi_x = calculate_susceptibility(vs, hi, direction="x")

        # Store results
        exact_magnetisation.append(exact_magnetisation_per_site)
        exact_gs_energiespersite.append(exact_gs_energypersite)
        energy_per_site.append(energy)
        magnetization_z.append(mz)
        magnetization_x.append(mx)
        susceptibility_z.append(chi_z)
        susceptibility_x.append(chi_x)
        Transverse.append(h)
        
        


    return Transverse, energy_per_site, magnetization_z, magnetization_x, susceptibility_z, susceptibility_x, exact_gs_energiespersite,exact_magnetisation

field_values = np.linspace(0, 2, 20)
N = 20
hi = nk.hilbert.Spin(s=1/2, N=N)
J = 1.0  # Coupling constant
for i in range(1,6):
    Transverse, energy_per_site, magnetization_z, magnetization_x, susceptibility_z, susceptibility_x, exact_gs_energiespersite,exact_magnetisation = run_vmc_for_field_strengths(field_values,hi,J)
    print('energypersite ',energy_per_site,exact_gs_energiespersite)
    np.savez(f"Ising data plot\\ising_run_{i}_with_mag{}.npz",Transverse=Transverse, energy_per_site=energy_per_site, magnetization_z=magnetization_z, magnetization_x=magnetization_x, susceptibility_z=susceptibility_z, susceptibility_x=susceptibility_x, exact_gs_energiespersite=exact_gs_energiespersite, exact_magnetisation=exact_magnetisation)
    print(f"data for run {i} , finished and saved data")
