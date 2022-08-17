#ifndef DECISION_TREE_CUH_
#define DECISION_TREE_CUH_

#include <stdint.h>
#include <thrust/sort.h>

template <typename T, typename B, typename D>
__global__ void ConstructDecisionTree(
    int32_t num_codebooks, int32_t num_samples, int32_t vector_length,
    int32_t dt_depth, const T* __restrict__ /* C x N x V */ samples,
    const B* __restrict__ /* C x N */ targets,
    D* __restrict__ /* C x (2 ^ dt_depth - 1) */ dims,
    T* __restrict__ /* C x (2 ^ dt_depth - 1) */ vals,
    B* __restrict__ /* C x (2 ^ dt_depth) */ bins) {
  int32_t codebook_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (codebook_idx >= num_codebooks) return;

  extern __shared__ char buffer[];
  int offset = 0;
  int* indices = (int*)(buffer + offset);               // C x N
  offset += num_codebooks * num_samples * sizeof(int);  //
  int* tmp_indices = (int*)(buffer + offset);           // C x N
  offset += num_codebooks * num_samples * sizeof(int);  //
  int* new_indices = (int*)(buffer + offset);           // C x N
  offset += num_codebooks * num_samples * sizeof(int);  //
  T* data = (T*)(buffer + offset);                      // C x N
  offset += num_codebooks * num_samples * sizeof(T);    //
  int* l_arr = (int*)(buffer + offset);                 // C x (2 ^ dt_depth)
  offset += num_codebooks * (1 << dt_depth) * sizeof(int);  //
  int* new_l_arr = (int*)(buffer + offset);  // C x (2 ^ dt_depth)

  int* indices_ptr = indices + codebook_idx * num_samples;
  int* new_indices_ptr = new_indices + codebook_idx * num_samples;
  int* tmp_indices_ptr = tmp_indices + codebook_idx * num_samples;
  T* data_ptr = data + codebook_idx * num_samples;
  int* l_ptr = l_arr + codebook_idx * (1 << dt_depth);
  int* new_l_ptr = new_l_arr + codebook_idx * (1 << dt_depth);

  const T* samples_ptr = samples + codebook_idx * num_samples * vector_length;
  const B* targets_ptr = targets + codebook_idx * num_samples;
  D* dims_ptr = dims + codebook_idx * ((1 << dt_depth) - 1);
  T* vals_ptr = vals + codebook_idx * ((1 << dt_depth) - 1);
  B* bins_ptr = bins + codebook_idx * (1 << dt_depth);

  for (int i = 0; i < num_samples; i++) indices_ptr[i] = i;
  l_ptr[0] = 0;

  for (int depth_idx = 0; depth_idx < dt_depth; depth_idx++) {
    for (int bin_idx = 0; bin_idx < (1 << depth_idx); bin_idx++) {
      int l = l_ptr[bin_idx];
      int r =
          bin_idx + 1 == (1 << depth_idx) ? num_samples : l_ptr[bin_idx + 1];
      int dim_idx = (1 << depth_idx) - 1 + bin_idx;
      if (r - l <= 1) {
        dims_ptr[dim_idx] = -1;
        new_l_ptr[bin_idx * 2] = l;
        new_l_ptr[bin_idx * 2 + 1] = r;
        continue;
      }

      float max_var = -1;
      for (int i = 0; i < vector_length; i++) {
        float sum = 0, sq_sum = 0;
        for (int j = l; j < r; j++) {
          data_ptr[j] = samples_ptr[indices_ptr[j] * vector_length + i];
          tmp_indices_ptr[j] = indices_ptr[j];
          sum += (float)data_ptr[j];
          sq_sum += (float)data_ptr[j] * (float)data_ptr[j];
        }
        float var = sq_sum - sum * sum / (r - l);
        if (var <= max_var) continue;

        max_var = var;
        dims_ptr[dim_idx] = i;
        thrust::sort_by_key(thrust::device, data_ptr + l, data_ptr + r,
                            tmp_indices_ptr + l);
        if ((r - l) % 2 == 0) {
          vals_ptr[dim_idx] =
              floor((data_ptr[(l + r) / 2 - 1] + data_ptr[(l + r) / 2]) / 2.0);
        } else {
          vals_ptr[dim_idx] = data_ptr[(l + r) / 2];
        }

        new_l_ptr[bin_idx * 2] = l;
        int j;
        for (j = l; j < r && data_ptr[j] <= vals_ptr[dim_idx]; j++)
          ;
        new_l_ptr[bin_idx * 2 + 1] = j;
        thrust::copy(thrust::device, tmp_indices_ptr + l, tmp_indices_ptr + r,
                     new_indices_ptr + l);
      }

      thrust::copy(thrust::device, new_indices_ptr + l, new_indices_ptr + r,
                   indices_ptr + l);
    }

    thrust::copy(thrust::device, new_l_ptr, new_l_ptr + num_samples, l_ptr);
  }

  for (int i = 0; i < 1 << dt_depth; i++) {
    bins_ptr[i] = targets_ptr[indices_ptr[l_ptr[i]]];
  }
}

#endif
