#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <string>
#include <tuple>

#include <adios2.h>

std::tuple<float, float, float> GenerateValues(size_t i, size_t j, size_t k, std::string mode)
{
    float x = static_cast<float>(i);
    float y = static_cast<float>(j);
    float z = static_cast<float>(k);
    if (mode == "linear")
        return {(6 * x * y) + (7 * z), (4 * x * z) + powf(y, 2), sqrtf(z) + (2 * x * y)};
    if (mode == "nonlinear")
        return {expf(2 * y) * sinf(x), sqrtf(z + 1) * cosf(x), powf(x, 2) * sinf(y) + (6 * z)};
    return {sinf(z), 4 * x, powf(y, 2) * cosf(x)};
}

void GenerateData(std::vector<float> &simArray1, std::vector<float> &simArray2, std::vector<float> &simArray3, size_t Nx, size_t Ny, size_t Nz, std::string mode)
{
    for (size_t i = 0; i < Nx; ++i)
    {
        for (size_t j = 0; j < Ny; ++j)
        {
            for (size_t k = 0; k < Nz; ++k)
            {
                size_t idx = (i * Ny * Nz) + (j * Nz) + k;
                auto value = GenerateValues(i, j, k, mode);
                simArray1[idx] = std::get<0>(value);
                simArray2[idx] = std::get<1>(value);
                simArray3[idx] = std::get<2>(value);
            }
        }
    }
}

void GenerateRandomData(std::vector<float> &simArray1, std::vector<float> &simArray2, std::vector<float> &simArray3)
{
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
    std::uniform_int_distribution<int> dist{1, 52};

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(simArray1.begin(), simArray1.end(), gen);
    std::generate(simArray2.begin(), simArray2.end(), gen);
    std::generate(simArray3.begin(), simArray3.end(), gen);
}

void AdiosTests(std::vector<float> simArray1, std::vector<float> simArray2, std::vector<float> simArray3, size_t Nx, size_t Ny, size_t Nz, std::string derivedVar) {
    adios2::ADIOS adios;
    adios2::IO bpOut = adios.DeclareIO("BPTestDerived");

    auto VX = bpOut.DefineVariable<float>("sim/VX", {Nx, Ny, Nz}, {0, 0, 0}, {Nx, Ny, Nz});
    auto VY = bpOut.DefineVariable<float>("sim/VY", {Nx, Ny, Nz}, {0, 0, 0}, {Nx, Ny, Nz});
    auto VZ = bpOut.DefineVariable<float>("sim/VZ", {Nx, Ny, Nz}, {0, 0, 0}, {Nx, Ny, Nz});

    // clang-format off
    if (derivedVar == "curl" || derivedVar == "all")
    {
        bpOut.DefineDerivedVariable("derived/curlV",
                                    "Vx =sim/VX \n"
                                    "Vy =sim/VY \n"
                                    "Vz =sim/VZ \n"
                                    "curl(Vx,Vy,Vz)",
                                    adios2::DerivedVarType::StoreData);
    }
    if (derivedVar == "magnitude" || derivedVar == "all")
    {
        bpOut.DefineDerivedVariable("derived/magV",
                                    "Vx =sim/VX \n"
                                    "Vy =sim/VY \n"
                                    "Vz =sim/VZ \n"
                                    "magnitude(Vx,Vy,Vz)",
                                    adios2::DerivedVarType::StoreData);
    }
    if (derivedVar == "add" || derivedVar == "all")
    {
        bpOut.DefineDerivedVariable("derived/addV",
                                    "Vx =sim/VX \n"
                                    "Vy =sim/VY \n"
                                    "Vz =sim/VZ \n"
                                    "Vx + Vy + Vz",
                                    adios2::DerivedVarType::StoreData);
    }
    // clang-format on
    std::string filename = "TestDerived.bp";
    adios2::Engine bpFileWriter = bpOut.Open(filename, adios2::Mode::Write);

    bpFileWriter.BeginStep();
    bpFileWriter.Put(VX, simArray1.data());
    bpFileWriter.Put(VY, simArray2.data());
    bpFileWriter.Put(VZ, simArray3.data());
    bpFileWriter.EndStep();
    bpFileWriter.Close();
}

float KernelAdd(std::vector<float> inputData[3],
              std::vector<float> &outValues) {
  auto start = std::chrono::steady_clock::now();
  size_t dataSize = inputData[0].size();
  for (size_t array=0; array<3; array++) {
	auto variable = inputData[array];
    for (size_t i = 0; i < dataSize; i++) {
      outValues[i] = outValues[i] + variable[i];
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> elapsed_seconds = end - start;
  return elapsed_seconds.count();
}

float KernelMagnitude(std::vector<float> inputData[3],
                    std::vector<float> &outValues) {
  auto start = std::chrono::steady_clock::now();
  size_t dataSize = inputData[0].size();
  for (size_t array=0; array<3; array++) {
	auto variable = inputData[array];
    for (size_t i = 0; i < dataSize; i++) {
      outValues[i] = outValues[i] + variable[i] * variable[i];
    }
  }
  for (size_t i = 0; i < dataSize; i++) {
    outValues[i] = std::sqrt(outValues[i]);
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> elapsed_seconds = end - start;
  return elapsed_seconds.count();
}

inline size_t returnIndex(size_t x, size_t y, size_t z, size_t dims[3]) {
  return z + y * dims[2] + x * dims[2] * dims[1];
}

double KernelCurl(std::vector<float> inputData[3], size_t dims[3],
               std::vector<float> &outValues) {
  auto start = std::chrono::steady_clock::now();
  size_t index = 0;
  for (int k = 0; k < dims[2]; ++k) {
    size_t next_k = std::max(0, k - 1),
           prev_k = std::min((int)dims[2] - 1, k + 1);
    for (int j = 0; j < dims[1]; ++j) {
      size_t next_j = std::max(0, j - 1),
             prev_j = std::min((int)dims[1] - 1, j + 1);
      for (int i = 0; i < dims[0]; ++i) {
        size_t next_i = std::max(0, i - 1),
               prev_i = std::min((int)dims[0] - 1, i + 1);
        // curl[0] = dv2 / dy - dv1 / dz
        outValues[3 * index] = (inputData[2][returnIndex(i, next_j, k, dims)] -
                                inputData[2][returnIndex(i, prev_j, k, dims)]) /
                               (next_j - prev_j);
        outValues[3 * index] +=
            (inputData[1][returnIndex(i, j, prev_k, dims)] -
             inputData[1][returnIndex(i, j, next_k, dims)]) /
            (next_k - prev_k);
        // curl[1] = dv0 / dz - dv2 / dx
        outValues[3 * index + 1] =
            (inputData[0][returnIndex(i, j, next_k, dims)] -
             inputData[0][returnIndex(i, j, prev_k, dims)]) /
            (next_k - prev_k);
        outValues[3 * index + 1] +=
            (inputData[2][returnIndex(prev_i, j, k, dims)] -
             inputData[2][returnIndex(next_i, j, k, dims)]) /
            (next_i - prev_i);
        // curl[2] = dv1 / dx - dv0 / dy
        outValues[3 * index + 2] =
            (inputData[1][returnIndex(next_i, j, k, dims)] -
             inputData[1][returnIndex(prev_i, j, k, dims)]) /
            (next_i - prev_i);
        outValues[3 * index + 2] +=
            (inputData[0][returnIndex(i, prev_j, k, dims)] -
             inputData[0][returnIndex(i, next_j, k, dims)]) /
            (next_j - prev_j);
        index++;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> elapsed_seconds = end - start;
  return elapsed_seconds.count();
}

void KernelTests(std::vector<float> simArray1, std::vector<float> simArray2, std::vector<float> simArray3, size_t Nx, size_t Ny, size_t Nz, std::string derivedVar) {
    size_t dims[3] = {Nx, Ny, Nz};
    size_t size = Nx * Ny * Nz;
    std::vector<float> arrayList[3] = {simArray1, simArray2, simArray3};
    float forceCompute = 0;

    if (derivedVar == "curl" || derivedVar == "all")
    {
        std::vector<float> destArray(3 * size);
        auto timeCurl = KernelCurl(arrayList, dims, destArray);
        forceCompute += destArray[0];
        std::cout << "CURL " << Nx << " " << size << " " << timeCurl << std::endl;
    }

    if (derivedVar == "magnitude" || derivedVar == "all")
    {
        std::vector<float> destArray(size);
        auto timeMag = KernelMagnitude(arrayList, destArray);
        forceCompute += destArray[0];
        std::cout << "MAG " << Nx << " " << size << " " << timeMag << std::endl;
    }

    if (derivedVar == "add" || derivedVar == "all")
    {
        std::vector<float> destArray(size);
        auto timeAdd = KernelAdd(arrayList, destArray);
        forceCompute += destArray[0];
        std::cout << "ADD " << Nx << " " << size << " " << timeAdd << std::endl;
    }
}

int main(int argc, char **argv) {
  size_t numVar = 3;
  std::string dataGenerationMode = "linear";
  std::vector<size_t> dim_list = {254, 309, 320, 358, 366, 374, 382, 389, 403};

  for (const size_t &dim : dim_list) {
    size_t Nx = dim, Ny = dim, Nz = dim;
    std::vector<float> simArray1(Nx * Ny * Nz);
    std::vector<float> simArray2(Nx * Ny * Nz);
    std::vector<float> simArray3(Nx * Ny * Nz);
    if (dataGenerationMode == "random")
        GenerateRandomData(simArray1, simArray2, simArray3);
    else
        GenerateData(simArray1, simArray2, simArray3, Nx, Ny, Nz, dataGenerationMode);
    
    std::cout << "Starting ADIOS tests for dimension " << dim << std::endl;
    AdiosTests(simArray1, simArray2, simArray3, Nx, Ny, Nz, "all");
    std::cout << "End ADIOS tests" << std::endl;
    std::cout << "Starting Kernel tests for dimension " << dim << std::endl;
    KernelTests(simArray1, simArray2, simArray3, Nx, Ny, Nz, "all");
    std::cout << "End Kernel tests" << std::endl;
  }

  return 0;
}
