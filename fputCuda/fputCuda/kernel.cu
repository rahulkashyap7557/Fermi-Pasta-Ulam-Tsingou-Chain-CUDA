
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>

#define NUM_THREADS 32
using namespace std;

// Declare CPU functions
void accel(double *x, double *a, int N, double alpha, double beta);
void verletStep1CPU(double *x, double *v, double *a, int N, double dt);
void verletStep2CPU(double *x, double *v, double *a, int N, double dt);

// Declare GPU functions
__global__ void accelGPU(double *x, double *a, int *N, double *alpha, double *beta);
__global__ void verletStep1GPU(double* x, double* v, double *a, int *N, double *dt);
__global__ void verletStep2GPU(double* x, double* v, double *a, int *N, double *dt);
__global__ void accelGPUModified(double* x, double* xLeft, double* xRight, double* a, int* N, double* alpha, double* beta);
__global__ void SetLeftAndRightArrays(double* x, double* xLeft, double* xRight, int* N);




int main()
{
	// Allocate memory
	int N = 100000;
	int nThreads = 16;
	int nBlocks = N / NUM_THREADS + 1;
	//int nBlocks = 1;
	double *x = new double[N];
	double *v = new double[N];
	double *a = new double[N];
	double* xLeft = new double[N];
	double* xRight = new double[N];
	double dt = 0.01;
	double alpha = 0.0;
	double beta = 1.0;

	// Initialize variables

	for (int i = 0; i < N; i++)
	{
		x[i] = 0.;
		xLeft[i] = 0.;
		xRight[i] = 0.;
		v[i] = 0.;
		a[i] = 0.;
		//cout << x[i] << '\t' << v[i] << '\t' << a[i] << endl;
	}

	//v[0] = 0.1;
	x[49] = 0.3;
	x[50] = -0.3;

	// Array for "left" element of x_i i.e. x_i-1
	for (int ii = 1; ii < N; ii++)
	{
		xLeft[ii] = x[ii - 1];
	}

	// Array for "right" element of x_i i.e. x_i+1
	for (int ii = 0; ii < N - 1; ii++)
	{
		xRight[ii] = x[ii + 1];
	}

	/*--------------------------------------------------------------------------------------
	GPU VERSION FOR COMPARISON
	---------------------------------------------------------------------------------------*/

	// Initialize device pointers
	double *d_alpha;
	double *d_beta;
	double *d_x;
	double* d_xLeft;
	double* d_xRight;
	double *d_v;
	double *d_a;
	double *d_dt;
	int *d_N;
	
	// Allocate memory on GPU
	cudaMalloc(&d_alpha, sizeof(double));
	cudaMalloc(&d_beta, sizeof(double));
	cudaMalloc(&d_dt, sizeof(double));
	cudaMalloc(&d_x, N * sizeof(double));
	cudaMalloc(&d_xLeft, N * sizeof(double));
	cudaMalloc(&d_xRight, N * sizeof(double));
	cudaMalloc(&d_v, N * sizeof(double));
	cudaMalloc(&d_a, N * sizeof(double));
	cudaMalloc(&d_N, sizeof(int));

	// Copy data to GPU
	cudaMemcpy(d_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dt, &dt, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_xLeft, xLeft, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_xRight, xRight, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);

	// Start the simulation
	int recSteps = 4000;
	int printInt = 1/dt;
	int T = recSteps * printInt;
	FILE *fp, *fp1, *fp2, *fp3, *fp5, *fp6, *fp7;

	fp = fopen("d_toten.dat", "w");
	fp1 = fopen("d_strsh.dat", "w");
	fp3 = fopen("d_velsh.dat", "w");
	fp5 = fopen("d_cmass.dat", "w");
	fp2 = fopen("d_restart.dat", "w");
	fp6 = fopen("d_ke.dat", "w");
	fp7 = fopen("d_acce.dat", "w");

	int writeT = 1;
	double cmass = 0.;
	int n;
	printf("Running on GPU \n");
	auto start = std::chrono::system_clock::now();
	for (int t = 0; t < T; t++)
	{
		//printf("writeT = %d\n", writeT);
		//accelGPU<<<1, N>>>(d_x, d_a, d_N, d_alpha);
		//accelGPU << <nBlocks, nThreads >> > (d_x, d_a, d_N, d_alpha, d_beta);
		verletStep1GPU<<<nBlocks, NUM_THREADS>>>(d_x, d_v, d_a, d_N, d_dt);
		accelGPUModified<<<nBlocks, NUM_THREADS>>>(d_x, d_xLeft, d_xRight, d_a, d_N, d_alpha, d_beta);
		//accel(x, a, N, alpha);
		verletStep2GPU<<<nBlocks, NUM_THREADS >>>(d_x, d_v, d_a, d_N, d_dt);
		SetLeftAndRightArrays<<<nBlocks, NUM_THREADS >>>(d_x, d_xLeft, d_xRight, d_N);

		// When write condition is met copy data back to CPU

		if (writeT == printInt)
		{
			//printf("Completed: %f%%\n", 100.*(t + 1) / T);
			//cudaMemcpy(&alpha, d_alpha, sizeof(double), cudaMemcpyDeviceToHost);
			//cudaMemcpy(&dt, d_dt, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(v, d_v, N * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(a, d_a, N * sizeof(double), cudaMemcpyDeviceToHost);
			//cudaMemcpy(&N, d_N, sizeof(int), cudaMemcpyDeviceToHost);

			for (n = 0; n < N; n++)
			{
				cmass += x[n];
				//printf("cmass=%f\n", cmass);
			}
			cmass = cmass / n;

			for (n = 0; n < N; n++)
			{
				//printf("here\n");
				//fprintf(fp1, "%.10f\t", x[n] - cmass);
				//fprintf(fp3, "%.10f\t", v[n]);
				//fprintf(fp7, "%.10f\t", a[n]);
			}
			//fprintf(fp1, "\n", x[n] - cmass);
			//fprintf(fp3, "\n", v[n]);
			//fprintf(fp7, "\n", a[n]);
			//fprintf(fp1, "\n");
			//fprintf(fp3, "\n");
			//fprintf(fp7, "\n");

			writeT = 0;
		}
		writeT++;

	}	

	fclose(fp); fclose(fp1); fclose(fp6); fclose(fp3); fclose(fp5); fclose(fp7);
	
	// Time the code

	auto end = std::chrono::system_clock::now();
	auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	std::cout << "time on GPU = " << elapsed << " milliseconds" << endl;
	
	/*--------------------------------------------------------------------------------------
	CPU VERSION FOR COMPARISON
	---------------------------------------------------------------------------------------*/
	
	for (int i = 0; i < N; i++)
	{
		x[i] = 0.;
		v[i] = 0.;
		a[i] = 0.;
		//cout << x[i] << '\t' << v[i] << '\t' << a[i] << endl;
	}
	//v[0] = 0.1;
	x[49] = 0.3;
	x[50] = -0.3;

	fp = fopen("toten.dat", "w");
	fp1 = fopen("strsh.dat", "w");
	fp3 = fopen("velsh.dat", "w");
	fp5 = fopen("cmass.dat", "w");
	fp2 = fopen("restart.dat", "w");
	fp6 = fopen("ke.dat", "w");
	fp7 = fopen("acce.dat", "w");

	printf("Running on CPU \n");
	start = std::chrono::system_clock::now();
	for (int t = 0; t < T; t++)
	{
		//accel(x, a, N, alpha);
		verletStep1CPU(x, v, a, N, dt);
		accel(x, a, N, alpha, beta);
		verletStep2CPU(x, v, a, N, dt);

		if (writeT == printInt)
		{
			//printf("Completed: %f%%\n", 100.*(t+1) / T);
			for (n = 0; n < N; n++)
			{
				cmass += x[n];
			}
			cmass = cmass / n;

			for (n = 0; n < N; n++)
			{
				//fprintf(fp1, "%.10f\t", x[n] - cmass);
				//fprintf(fp3, "%.10f\t", v[n]);
				//fprintf(fp7, "%.10f\t", a[n]);
			}
			//fprintf(fp1, "\n", x[n] - cmass);
			//fprintf(fp3, "\n", v[n]);
			//fprintf(fp7, "\n", a[n]);
			//fprintf(fp1, "\n");
			//fprintf(fp3, "\n");
			//fprintf(fp7, "\n");

			writeT = 0;
		}
		writeT++;
		
	}

	fclose(fp); fclose(fp1); fclose(fp6); fclose(fp3); fclose(fp5); fclose(fp7);

	// Time the code

	end = std::chrono::system_clock::now();
	elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	std::cout << "time on CPU = " << elapsed << " milliseconds" << endl;

	// Print results
	

	// Free memory
	delete[] x;
	delete[] v;
	delete[] a;

	cudaFree(d_a);
	cudaFree(d_x);
	cudaFree(d_xLeft);
	cudaFree(d_xRight);
	cudaFree(d_v);
	cudaFree(d_dt);
	cudaFree(d_N);
	cudaFree(d_alpha);

	return 0;
}

void accel(double *x, double *a, int N, double alpha, double beta)
{
	int i, j, k;
	double xi;
	double xj, xk;
	double fac;
	double fac1, fac2;
	double fac13, fac23;
	
	//int N = sizeof(a) / sizeof(a[0]);
	for (i = 0; i < N; i++)
	{
		xi = x[i];
		j = i - 1;
		k = i + 1;

		if (j == -1)
		{
			xj = 0.0;
		}
		else
		{
			xj = x[j];
		}

		if (k == N)
		{
			xk = 0.0;
		}
		else
		{
			xk = x[k];
		}

		fac = (2.0 * xi) - xj - xk;
		fac1 = xk - xi;
		fac2 = xi - xj;
		fac13 = fac1 * fac1 * fac1;
		fac23 = fac2 * fac2 * fac2;


		a[i] = -2.0*alpha*fac + 4.0*beta*(fac13 - fac23);	
		//cout << a[i] << endl;
		

	}
}


void verletStep1CPU(double* x, double* v, double *a, int N, double dt)
{
	//int N = sizeof(a) / sizeof(a[0]);
	double dt2 = dt * dt;

	for (int i = 0; i < N; i++)
	{
		x[i] = x[i] + v[i] * dt + 0.5*a[i] * dt2;
		v[i] += 0.5 * a[i] * dt;
	}

}

void verletStep2CPU(double* x, double* v, double *a, int N, double dt)
{
	//int N = sizeof(a) / sizeof(a[0]);
	double dt2 = dt * dt;

	for (int i = 0; i < N; i++)
	{
		v[i] += 0.5*a[i] * dt;
	}

}

__global__ void accelGPU(double *x, double *a, int *N, double *alpha, double *beta)
{
	int i, j, k;
	double xi;
	double xj, xk;
	double fac;
	double fac1, fac2;
	double fac13, fac23;
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	//int idx = threadIdx.x;
	//printf("N = %d, alpha = %f", *N, *alpha);
	j = idx - 1;
	k = idx + 1;

	if (idx < *N);
	{
		xi = x[idx];		

		if (j == -1)
		{
			xj = 0.0;
		}
		else
		{
			xj = x[j];
		}

		if (k == *N)
		{
			xk = 0.0;
		}
		else
		{
			xk = x[k];
		}

		fac = (2.0 * xi) - xj - xk;
		fac1 = xk - xi;
		fac2 = xi - xj;
		fac13 = fac1 * fac1 * fac1;
		fac23 = fac2 * fac2 * fac2;

		a[idx] = -2.0*(*alpha)*fac + 4.0*(*beta)*(fac13 - fac23);
		//printf("%f", a[idx]);
		//printf("\n");
		
	}

}

__global__ void verletStep1GPU(double* x, double* v, double *a, int *N, double *dt)
{
	//int N = sizeof(a) / sizeof(a[0]);
	double dt2 = (*dt) * (*dt);
	int idx = idx = threadIdx.x + blockDim.x*blockIdx.x;
	//int idx = threadIdx.x;
	//printf("idx = %d \n", idx);
	if (idx < *N)
	{
		x[idx] = x[idx] + v[idx] * (*dt) + 0.5*a[idx] * dt2;
		v[idx] += 0.5 * a[idx] * (*dt);		
	}
	

}

__global__ void verletStep2GPU(double* x, double* v, double *a, int *N, double *dt)
{
	//int N = sizeof(a) / sizeof(a[0]);
	int idx = idx = threadIdx.x + blockDim.x*blockIdx.x;
	//int idx = threadIdx.x;
	if (idx < *N)
	{
		v[idx] += 0.5*a[idx] * (*dt);
	}

}

__global__ void accelGPUModified(double* x, double* xLeft, double* xRight, double* a, int* N, double* alpha, double* beta)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	////printf("N = %d, alpha = %f", *N, *alpha);

	if (idx < *N);
	{
		//const int tileSize = blockDim.x;

		// Hate hardcoing, but shared mem cannot be initialized using dynamic variable.
		// For instance, trying to say xl[tileSize] doesn't work unless tileSize is known at compile time.
		// Update: We can declare extern and do this ^^. Needs to be changed
		__shared__ double xLeftShared[NUM_THREADS];
		__shared__ double xRightShared[NUM_THREADS];
		__shared__ double xShared[NUM_THREADS];

		// Retrieve the correct values of the shared memories from global memory
		
		int threadId = threadIdx.x;

		// Load shared memory when the thread id in the block is 0.
		// This should be updated so that all threads collaboratively load memory. 
		// For now, let us load using a single thread only.
		if (threadId == 0)
		{
			// Load shared memory
			for (int ii = 0; ii < NUM_THREADS; ii++)
			{
				xShared[ii] = x[idx + ii];
				xLeftShared[ii] = xLeft[idx + ii];
				xRightShared[ii] = xRight[idx + ii];
			}

		}

		// Once they are retrieved, calculate the acceleration
		int i, j, k;
		double xi;
		double xj, xk;
		double fac;
		double fac1, fac2;
		double fac13, fac23;
		double ai;

		xi = xShared[idx];
		xj = xLeftShared[idx];
		xk = xRightShared[idx];

		fac = (2.0 * xi) - xj - xk;
		fac1 = xk - xi;
		fac2 = xi - xj;
		fac13 = fac1 * fac1 * fac1;
		fac23 = fac2 * fac2 * fac2;

		ai = -2.0 * (*alpha) * fac + 4.0 * (*beta) * (fac13 - fac23);
		a[idx] = a[i];
	}
}

__global__ void SetLeftAndRightArrays(double* x, double* xLeft, double* xRight, int* N)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx > 0 && idx < *N)
	{		
		xLeft[idx] = x[idx - 1];
	}
	else if (idx == 0)
	{
		xLeft[idx] = 0.;
	}

	if (idx < *N - 1)
	{
		xRight[idx] = x[idx + 1];
	}
	else if (idx == *N - 1)
	{
		xRight[idx] = 0.;
	}
}
