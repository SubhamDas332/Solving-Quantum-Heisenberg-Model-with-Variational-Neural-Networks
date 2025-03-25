import os
import ctypes
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
# Prevent the system from entering sleep mode
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
# Prevent sleep
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)



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

def run_vmc_for_field_strengths(field_values, hi, J):

    energy_per_site = []
    magnetization_z = []
    magnetization_x = []
    susceptibility_z = []
    susceptibility_x = []
    exact_gs_energiespersite=[]
    seed=[]
    Transverse=[]

    for h in field_values:

        H = transverse_field_heisenberg_hamiltonian(hi.size, J, h)
        evals = nk.exact.lanczos_ed(H, compute_eigenvectors=False)
        exact_gs_energypersite = evals[0]/hi.size # exact_gs_energy = -39.14752260706246
                                            # exact_gs_energy = -34.72989333759587  #N=20
                                            # exact_gs_energy =  -17.032140829131475
        # print('exact ground state',exact_gs_energypersite)

        ranint=np.random.randint(0,999999)
        ffnn = Model()
        sa =  nk.sampler.MetropolisLocal(hilbert=hi)
        op = nk.optimizer.Sgd(learning_rate=0.05) #,nesterov=True
        sr = nk.optimizer.SR(diag_shift=0.1,holomorphic=True)
        vs = nk.vqs.MCState(sa, ffnn, n_samples=2048,seed=ranint,sampler_seed=ranint) #seed=ranint, sampler_seed=ranint
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
        exact_gs_energiespersite.append(exact_gs_energypersite)
        energy_per_site.append(energy)
        magnetization_z.append(mz)
        magnetization_x.append(mx)
        susceptibility_z.append(chi_z)
        susceptibility_x.append(chi_x)
        seed.append(ranint)
        Transverse.append(h)
        
        #save data
        # np.savez(f"Heisenberg_run_{h}.npz",Transverse=Transverse,energy_per_site=energy_per_site, magnetization_z=magnetization_z, magnetization_x=magnetization_x, susceptibility_z=susceptibility_z, susceptibility_x=susceptibility_x, exact_gs_energiespersite=exact_gs_energiespersite,seed=seed )

    return Transverse,energy_per_site, magnetization_z, magnetization_x, susceptibility_z, susceptibility_x, exact_gs_energiespersite, seed



field_values = np.linspace(0, 7.5, 20)
N = 20
hi = nk.hilbert.Spin(s=1/2, N=N)
J = 1.0  

for i in range(0,20):
# Coupling constant
    Transverse, energy_per_site, magnetization_z, magnetization_x, susceptibility_z, susceptibility_x, exact_gs_energiespersite,seed = run_vmc_for_field_strengths(field_values,hi,J)
    print('energypersite',energy_per_site)

    np.savez(f"code\\Heisenberg_latest_plot_data\\run_{i}.npz",Transverse=Transverse,energy_per_site=energy_per_site, magnetization_z=magnetization_z, magnetization_x=magnetization_x, susceptibility_z=susceptibility_z, susceptibility_x=susceptibility_x, exact_gs_energiespersite=exact_gs_energiespersite,seed=seed )
    print(f"run number{i} finished and saved data")


# Allow sleep again
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)