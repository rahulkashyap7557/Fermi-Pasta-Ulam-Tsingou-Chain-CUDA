# Fermi-Pasta-Ulam-Tsingou-Chain-CUDA
CUDA version of interacting nonlinear many-body systems. This is a nonlinear chain with nearest neighbor interactions. If successful, it will be integrated into PULSEDYN.

Current code uses global memory and is slower than serial version on CPU (typically) unless chain length is very large. Future updates include leveraging shared and texture memories to speed up code. 
