# Fermi-Pasta-Ulam-Tsingou-Chain-CUDA
CUDA version of interacting nonlinear many-body systems. This is a nonlinear chain with nearest neighbor interactions. If successful, it will be integrated into PULSEDYN.

Solves equations of motion with the Hamiltonian

$$H = \Sigma_{i=1}^{N} \frac{p_i^2}{2m} + \Sigma_{i=0}^{N+1}V(x_{i+1} - x_i)$$.

Here, 

$$V(x_{i+1} - x_i) = \alpha*(x_{i+1} - x_i)^2 + \beta(x_{i+1} - x_i)^4.$$

The boundaries are reflecting.

Current code uses global memory and is slower than serial version on CPU (typically) unless chain length is very large. Future updates include leveraging shared and texture memories to speed up code. 

Compile using

`nvcc kernel.cu -o kernel -arch=sm_##`

In place of ##, use the compute capability of your card. Ex: For compute capability of 3.5, use -arch=sm_35
