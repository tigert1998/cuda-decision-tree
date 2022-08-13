#include <vector>

#include "decision_tree.cuh"

int main() {
  int8_t* samples_ptr;
  int8_t* targets_ptr;
  int8_t* dims_ptr;
  int8_t* vals_ptr;
  int8_t* bins_ptr;

  int num_codebooks, num_samples, vector_length, dt_depth;

  std::cin >> num_codebooks >> num_samples >> vector_length >> dt_depth;

  std::vector<int8_t> samples(num_codebooks * num_samples * vector_length);
  std::vector<int8_t> targets(num_codebooks * num_samples);

  for (int i = 0; i < samples.size(); i++) {
    int x;
    std::cin >> x;
    samples[i] = x;
  }
  for (int i = 0; i < targets.size(); i++) {
    int x;
    std::cin >> x;
    targets[i] = i;
  }

  cudaMalloc(&samples_ptr, num_codebooks * num_samples * vector_length);
  cudaMalloc(&targets_ptr, num_codebooks * num_samples);
  cudaMalloc(&dims_ptr, num_codebooks * ((1 << dt_depth) - 1));
  cudaMalloc(&vals_ptr, num_codebooks * ((1 << dt_depth) - 1));
  cudaMalloc(&bins_ptr, num_codebooks * (1 << dt_depth));

  cudaMemcpy(samples_ptr, samples.data(), samples.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(targets_ptr, targets.data(), targets.size(),
             cudaMemcpyHostToDevice);

  int shared_memory_size = num_codebooks * num_samples * 3 * sizeof(int) +
                           num_codebooks * num_samples * sizeof(int8_t) +
                           num_codebooks * (1 << dt_depth) * 2 * sizeof(int);
  ConstructDecisionTree<int8_t, int8_t, int8_t>
      <<<num_codebooks, 1, shared_memory_size>>>(
          num_codebooks, num_samples, vector_length, dt_depth, samples_ptr,
          targets_ptr, dims_ptr, vals_ptr, bins_ptr);

  std::vector<int8_t> dims(num_codebooks * ((1 << dt_depth) - 1)),
      vals(num_codebooks * ((1 << dt_depth) - 1)),
      bins(num_codebooks * (1 << dt_depth));

  cudaMemcpy(dims.data(), dims_ptr, dims.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(vals.data(), vals_ptr, vals.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(bins.data(), bins_ptr, bins.size(), cudaMemcpyDeviceToHost);

  for (int i = 0; i < dims.size(); i++) printf("%d ", dims[i]);
  puts("");
  for (int i = 0; i < vals.size(); i++) printf("%d ", vals[i]);
  puts("");
  for (int i = 0; i < bins.size(); i++) printf("%d ", bins[i]);
  puts("");

  return 0;
}