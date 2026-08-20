
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <cuda_runtime.h>

// Explicitly define M_PI to ensure it's a compile-time constant for __constant__ variables
#define M_PI 3.14159265358979323846
#define BLOCK_SIZE 256

// Structure of arrays
struct Particles
{
  float* position_x;
  float* position_y;
  float* position_z;
  float* velocity_x;
  float* velocity_y;
  float* velocity_z;
  // Pointers for trajectory storage
  float* d_pos_x;
  float* d_pos_y;
  float* d_pos_z;
};

// Define constants for device code using #define for compile-time substitution
// D_Q increased significantly to make particle movement visible for simulation purposes
#define D_Q 1.6e-19
#define D_M 1.67e-27
#define D_PERMITIVITY 8.854e-12
#define D_EPSILON2 1e-12 // Increased epsilon2 for better stability (softening length of 1mm)
#define d_coulombCoeff (D_Q * D_Q) / (4.0 * M_PI * D_PERMITIVITY * D_M)

__device__ void prodVect(const float* a, const float* b, float* res)
{
  res[0] = a[1] * b[2] - a[2] * b[1];
  res[1] = a[2] * b[0] - a[0] * b[2];
  res[2] = a[0] * b[1] - a[1] * b[0];
}

struct rk4_parameters
{
  struct Particles* particles;
  int nb_particles;
  float t;
  float* state;
  int stateSize;
  float dt;
  float* dydt;
  float* shared_pos_x;
  float* shared_pos_y;
  float* shared_pos_z;
};

__device__ void compute_dydt(struct rk4_parameters* params, float* currentState, float* dydt)
{

  float electrical_field[3] = {0, 0, 0};
  float magnetical_field[3] = {0, 0, 1e-2}; // Champ B non nul pour observer une trajectoire

  float velocity[3] = { currentState[3], currentState[4], currentState[5] };
  float acceleration[3];

  float prodVect_temp[3] = {0, 0, 0};
  prodVect(&currentState[3], magnetical_field, prodVect_temp);

  // Use #defined constants
  acceleration[0] = (D_Q/D_M) * (electrical_field[0] + prodVect_temp[0]);
  acceleration[1] = (D_Q/D_M) * (electrical_field[1] + prodVect_temp[1]);
  acceleration[2] = (D_Q/D_M) * (electrical_field[2] + prodVect_temp[2]);

  int numTiles = (params->nb_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int numTile = 0; numTile < numTiles; numTile++)
  {

    int particle_to_load = numTile * BLOCK_SIZE + threadIdx.x;

    if (particle_to_load < params->nb_particles)
    {
      params->shared_pos_x[threadIdx.x] = params->particles->position_x[particle_to_load];
      params->shared_pos_y[threadIdx.x] = params->particles->position_y[particle_to_load];
      params->shared_pos_z[threadIdx.x] = params->particles->position_z[particle_to_load];
    }
    else
    {
      params->shared_pos_x[threadIdx.x] = 0.0;
      params->shared_pos_y[threadIdx.x] = 0.0;
      params->shared_pos_z[threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Limiter la boucle aux particules réellement valides de la tuile
    int limit_j = min(BLOCK_SIZE, params->nb_particles - numTile * BLOCK_SIZE);

    for (int j = 0; j < limit_j; j++)
    {
      // S'assurer que l'on ne calcule pas la force avec la particule elle meme
      if (numTile * BLOCK_SIZE + j == blockIdx.x * blockDim.x + threadIdx.x) continue;

      float dx = (currentState[0] - params->shared_pos_x[j]);
      float dy = (currentState[1] - params->shared_pos_y[j]);
      float dz = (currentState[2] - params->shared_pos_z[j]);

      float dist2 = dx * dx + dy * dy + dz * dz + D_EPSILON2;
      float inv_dist = rsqrtf(dist2);
      float inv_dist3 = inv_dist * inv_dist * inv_dist;

      float force_factor = d_coulombCoeff * inv_dist3;

      acceleration[0] += dx * force_factor;
      acceleration[1] += dy * force_factor;
      acceleration[2] += dz * force_factor;
    }

    __syncthreads();
  }

  dydt[0] = velocity[0];
  dydt[1] = velocity[1];
  dydt[2] = velocity[2];
  dydt[3] = acceleration[0];
  dydt[4] = acceleration[1];
  dydt[5] = acceleration[2];
}

__device__ void step_rk4(struct rk4_parameters* params)
{
  float k[4][6];
  float tmp[6];
  // Initialize k and tmp to 0. It is good practice.
  for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 6; ++j) {
          k[i][j] = 0.0;
      }
  }
  for (int i = 0; i < 6; ++i) {
      tmp[i] = 0.0;
  }

  for (int i = 0; i < params->stateSize; ++i)
    tmp[i] = params->state[i];
  compute_dydt(params, tmp, k[0]);

  for (int i = 0; i < params->stateSize; ++i)
    tmp[i] = params->state[i] + (params->dt / 2.0) * k[0][i];
  compute_dydt(params, tmp, k[1]);

  for (int i = 0; i < params->stateSize; ++i)
    tmp[i] = params->state[i] + (params->dt / 2.0) * k[1][i];
  compute_dydt(params, tmp, k[2]);

  for (int i = 0; i < params->stateSize; ++i)
    tmp[i] = params->state[i] + params->dt * k[2][i];
  compute_dydt(params, tmp, k[3]);

  for (int i = 0; i < params->stateSize; i++)
    params->state[i] += (params->dt / 6.0) * (k[0][i] + 2.0 * k[1][i] + 2.0 * k[2][i] + k[3][i]);
}

__global__ void update_particles(struct Particles particles, const int nb_particles, int step, float dt) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb_step = 10000;

    float state[6] = {0};
    float dydt[6] = {0};

    // Gestion de la mémoire partagée
    __shared__ float pos_x[BLOCK_SIZE];
    __shared__ float pos_y[BLOCK_SIZE];
    __shared__ float pos_z[BLOCK_SIZE];

    struct rk4_parameters kernel_params;
    kernel_params.state = state;
    kernel_params.dydt = dydt;

    kernel_params.shared_pos_x = pos_x;
    kernel_params.shared_pos_y = pos_y;
    kernel_params.shared_pos_z = pos_z;

    if (idx < nb_particles)
    {
      kernel_params.state[0] = particles.position_x[idx];
      kernel_params.state[1] = particles.position_y[idx];
      kernel_params.state[2] = particles.position_z[idx];
      kernel_params.state[3] = particles.velocity_x[idx];
      kernel_params.state[4] = particles.velocity_y[idx];
      kernel_params.state[5] = particles.velocity_z[idx];
    }

    kernel_params.nb_particles = nb_particles;
    kernel_params.t = (float)step * dt;
    kernel_params.stateSize = 6;
    kernel_params.dt = dt;
    kernel_params.particles = &particles;

    step_rk4(&kernel_params);

    if (idx < nb_particles)
    {
      particles.position_x[idx] = kernel_params.state[0];
      particles.position_y[idx] = kernel_params.state[1];
      particles.position_z[idx] = kernel_params.state[2];
      particles.velocity_x[idx] = kernel_params.state[3];
      particles.velocity_y[idx] = kernel_params.state[4];
      particles.velocity_z[idx] = kernel_params.state[5];
    }

    if (idx < 10)
    {
      int trajectory_linear_index = idx * nb_step + step;

      particles.d_pos_x[trajectory_linear_index] = kernel_params.state[0];
      particles.d_pos_y[trajectory_linear_index] = kernel_params.state[1];
      particles.d_pos_z[trajectory_linear_index] = kernel_params.state[2];
    }

} // Closing brace for __global__ void update_particles

int main(int argc, char* argv[])
{
  const float dt = 1e-9;
  const int k_nb_particles = 2000;
  int nb_step = 10000;
  int nb_particules_result = 10;

  struct Particles h_particles;
  h_particles.position_x = new float[k_nb_particles];
  h_particles.position_y = new float[k_nb_particles];
  h_particles.position_z = new float[k_nb_particles];
  h_particles.velocity_x = new float[k_nb_particles];
  h_particles.velocity_y = new float[k_nb_particles];
  h_particles.velocity_z = new float[k_nb_particles];

  struct Particles d_particles;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distPosition(-1e-4, 1e-4); // Meters
  std::uniform_real_distribution<float> distVelocity(-1e3, 1e3); // Meters/second

  for (int i = 0; i < k_nb_particles; i++)
  {
      h_particles.position_x[i] = distPosition(gen);
      h_particles.position_y[i] = distPosition(gen);
      h_particles.position_z[i] = distPosition(gen);
      h_particles.velocity_x[i] = distVelocity(gen);
      h_particles.velocity_y[i] = distVelocity(gen);
      h_particles.velocity_z[i] = distVelocity(gen);
  }

  struct timespec debut;
  struct timespec fin;
  clock_gettime(CLOCK_MONOTONIC, &debut);

  cudaMalloc(&d_particles.position_x, k_nb_particles*sizeof(float));
  cudaMalloc(&d_particles.position_y, k_nb_particles*sizeof(float));
  cudaMalloc(&d_particles.position_z, k_nb_particles*sizeof(float));
  cudaMalloc(&d_particles.velocity_x, k_nb_particles*sizeof(float));
  cudaMalloc(&d_particles.velocity_y, k_nb_particles*sizeof(float));
  cudaMalloc(&d_particles.velocity_z, k_nb_particles*sizeof(float));

  size_t bytes = k_nb_particles * sizeof(float);

  cudaMemcpy(d_particles.position_x, h_particles.position_x, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_particles.position_y, h_particles.position_y, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_particles.position_z, h_particles.position_z, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_particles.velocity_x, h_particles.velocity_x, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_particles.velocity_y, h_particles.velocity_y, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_particles.velocity_z, h_particles.velocity_z, bytes, cudaMemcpyHostToDevice);

  cudaMalloc(&d_particles.d_pos_x, nb_particules_result*nb_step*sizeof(float));
  cudaMalloc(&d_particles.d_pos_y, nb_particules_result*nb_step*sizeof(float));
  cudaMalloc(&d_particles.d_pos_z, nb_particules_result*nb_step*sizeof(float));

  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid = (k_nb_particles + threadsPerBlock - 1) / threadsPerBlock;

  for (int step_rk4 = 0; step_rk4 < nb_step; step_rk4++)
  {
    update_particles<<<blocksPerGrid, threadsPerBlock>>>(d_particles, k_nb_particles, step_rk4, dt);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Erreur de GPU : %s\n", cudaGetErrorString(err));
      break;
    }
  }

  cudaDeviceSynchronize();

  float* h_trajectory_x = new float[nb_particules_result * nb_step];
  float* h_trajectory_y = new float[nb_particules_result * nb_step];
  float* h_trajectory_z = new float[nb_particules_result * nb_step];
  cudaMemcpy(h_trajectory_x, d_particles.d_pos_x, nb_particules_result * nb_step * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_trajectory_y, d_particles.d_pos_y, nb_particules_result * nb_step * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_trajectory_z, d_particles.d_pos_z, nb_particules_result * nb_step * sizeof(float), cudaMemcpyDeviceToHost);

  std::ofstream outputFile("particle_trajectory.csv");
  if (!outputFile.is_open()) {
      std::cerr << "Error: Could not open file particle_trajectory.csv" << std::endl;
      return 1;
  }

  outputFile << "x,y,z\n";

  for (int i = 0; i < nb_particules_result; i++)
  {
    for (int step=0; step<nb_step; step++)
    {
      outputFile << h_trajectory_x[i*nb_step + step] << "," << h_trajectory_y[i*nb_step + step] << "," << h_trajectory_z[i*nb_step + step] << "\n";
    }
  }

  // Close the file
  outputFile.close();

  cudaMemcpy(h_particles.position_x, d_particles.position_x, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_particles.position_y, d_particles.position_y, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_particles.position_z, d_particles.position_z, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_particles.velocity_x, d_particles.velocity_x, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_particles.velocity_y, d_particles.velocity_y, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_particles.velocity_z, d_particles.velocity_z, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_particles.position_x);
  cudaFree(d_particles.position_y);
  cudaFree(d_particles.position_z);
  cudaFree(d_particles.velocity_x);
  cudaFree(d_particles.velocity_y);
  cudaFree(d_particles.velocity_z);

  cudaFree(d_particles.d_pos_x);
  cudaFree(d_particles.d_pos_y);
  cudaFree(d_particles.d_pos_z);

  delete[] h_particles.position_x;
  delete[] h_particles.position_y;
  delete[] h_particles.position_z;
  delete[] h_particles.velocity_x;
  delete[] h_particles.velocity_y;
  delete[] h_particles.velocity_z;

  delete [] h_trajectory_x;
  delete [] h_trajectory_y;
  delete [] h_trajectory_z;

  clock_gettime(CLOCK_MONOTONIC, &fin);

  long secondes = fin.tv_sec - debut.tv_sec;
  long nanosecondes = fin.tv_nsec - debut.tv_nsec;
  long total_ns = secondes * 1000000000 + nanosecondes;

  printf("Temps en secondes : %f s\n", total_ns / 1e9);

  return 0;
}
