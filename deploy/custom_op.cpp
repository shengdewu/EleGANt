#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <torch/script.h>


torch::Tensor linalg_pinv(const torch::Tensor &input){
    return at::linalg_pinv(input, 1e-15, false);
}

/// \param input
/// \param grid
/// \param interpolation_mode 0-bilinear, 1-nearest, 2-bicubic
/// \param padding_mode 0-zeros, 1-border, 2-reflection
/// \param align_corners
torch::Tensor grid_sampler(const torch::Tensor &input, const torch::Tensor &grid, int64_t mode=0, int64_t padding_mode=0, bool align_corners=false){
    return at::grid_sampler(input, grid, mode, padding_mode, align_corners);
}

//域名要和 register_custom_op_symbolic("custom::linalg_pinv") 一致
static auto registry_pinv = torch::RegisterOperators("custom::linalg_pinv", &linalg_pinv);
static auto registry_grid = torch::RegisterOperators("custom::grid_sampler", &grid_sampler);
