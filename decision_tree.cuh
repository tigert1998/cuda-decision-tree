#ifndef DECISION_TREE_CUH_
#define DECISION_TREE_CUH_

#include <stdint.h>
#include <thrust/sort.h>

template <typename T, typename B, typename D>
__device__ float SelectBestSplit(int32_t l, int32_t r, int32_t dim,
                                 int32_t num_targets, int32_t vector_length,
                                 const T* samples_ptr, const B* targets_ptr,
                                 const int* indices_ptr, T* data_ptr,
                                 int* tmp_indices_ptr, int* l_targets_cnt_ptr,
                                 int* r_targets_cnt_ptr, int* split) {
  // returns minimum gini index

  thrust::fill(thrust::device, l_targets_cnt_ptr,
               l_targets_cnt_ptr + num_targets, 0);
  thrust::fill(thrust::device, r_targets_cnt_ptr,
               r_targets_cnt_ptr + num_targets, 0);
  for (int i = l; i < r; i++) {
    data_ptr[i] = samples_ptr[indices_ptr[i] * vector_length + dim];
    tmp_indices_ptr[i] = indices_ptr[i];
    r_targets_cnt_ptr[targets_ptr[tmp_indices_ptr[i]]]++;
  }

  thrust::sort_by_key(thrust::device, data_ptr + l, data_ptr + r,
                      tmp_indices_ptr + l);

  float min_gini = 1e9;

  for (int i = l; i < r - 1; i++) {
    // [l, i] [i + 1, r - 1]
    l_targets_cnt_ptr[targets_ptr[tmp_indices_ptr[i]]]++;
    r_targets_cnt_ptr[targets_ptr[tmp_indices_ptr[i]]]--;

    float l_gini = 1, r_gini = 1;
    for (int j = 0; j < num_targets; j++) {
      float p = l_targets_cnt_ptr[j] * 1.0 / (i - l + 1);
      l_gini -= p * p;
    }
    for (int j = 0; j < num_targets; j++) {
      float p = r_targets_cnt_ptr[j] * 1.0 / (r - 1 - i);
      r_gini -= p * p;
    }

    if (l_gini + r_gini < min_gini) {
      min_gini = l_gini + r_gini;
      *split = i + 1;
    }
  }

  return min_gini;
}

template <typename T, typename B, typename D>
__global__ void ConstructDecisionTree(
    int32_t num_codebooks, int32_t num_samples, int32_t vector_length,
    int32_t num_targets, int32_t dt_depth,
    const T* __restrict__ /* C x N x V */ samples,
    const B* __restrict__ /* C x N */ targets,
    D* __restrict__ /* C x (2 ^ dt_depth - 1) */ dims,
    T* __restrict__ /* C x (2 ^ dt_depth - 1) */ vals,
    B* __restrict__ /* C x (2 ^ dt_depth) */ bins) {
  int32_t codebook_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (codebook_idx >= num_codebooks) return;

  int* indices_ptr = new int[num_samples];
  int* new_indices_ptr = new int[num_samples];
  int* tmp_indices_ptr = new int[num_samples];
  T* data_ptr = new T[num_samples];
  int* l_ptr = new int[1 << dt_depth];
  int* new_l_ptr = new int[1 << dt_depth];
  int* l_targets_cnt_ptr = new int[num_targets];
  int* r_targets_cnt_ptr = new int[num_targets];

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

      float min_gini = 1e9;
      for (int i = 0; i < vector_length; i++) {
        int split;
        float gini = SelectBestSplit<T, B, D>(
            l, r, i, num_targets, vector_length, samples_ptr, targets_ptr,
            indices_ptr, data_ptr, tmp_indices_ptr, l_targets_cnt_ptr,
            r_targets_cnt_ptr, &split);
        if (gini >= min_gini) continue;

        dims_ptr[dim_idx] = i;
        new_l_ptr[bin_idx * 2] = l;
        new_l_ptr[bin_idx * 2 + 1] = split;
        vals_ptr[dim_idx] =
            floor((data_ptr[split - 1] + data_ptr[split]) / 2.0);
        thrust::copy(thrust::device, tmp_indices_ptr + l, tmp_indices_ptr + r,
                     new_indices_ptr + l);
      }

      thrust::copy(thrust::device, new_indices_ptr + l, new_indices_ptr + r,
                   indices_ptr + l);
    }

    thrust::copy(thrust::device, new_l_ptr, new_l_ptr + num_samples, l_ptr);
  }

  for (int i = 0; i < 1 << dt_depth; i++) {
    int l = l_ptr[i];
    int r = i + 1 == (1 << dt_depth) ? num_samples : l_ptr[i + 1];

    if (l == r) {
      bins_ptr[i] = targets_ptr[indices_ptr[l]];
    } else {
      thrust::fill(thrust::device, l_targets_cnt_ptr,
                   l_targets_cnt_ptr + num_targets, 0);
      for (int j = l; j < r; j++) {
        l_targets_cnt_ptr[targets_ptr[indices_ptr[j]]]++;
      }
      int max_num = l_targets_cnt_ptr[0];
      bins_ptr[i] = 0;
      for (int j = 1; j < num_targets; j++) {
        if (l_targets_cnt_ptr[j] > max_num) {
          max_num = l_targets_cnt_ptr[j];
          bins_ptr[i] = j;
        }
      }
    }
  }

  delete[] indices_ptr;
  delete[] new_indices_ptr;
  delete[] tmp_indices_ptr;
  delete[] data_ptr;
  delete[] l_ptr;
  delete[] new_l_ptr;
  delete[] l_targets_cnt_ptr;
  delete[] r_targets_cnt_ptr;
}

#endif
