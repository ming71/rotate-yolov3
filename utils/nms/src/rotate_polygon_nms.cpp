#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);

at::Tensor r_nms(const at::Tensor& dets, const float threshold) {
  CHECK_CUDA(dets);
  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  return nms_cuda(dets, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("r_nms", &r_nms, "r_nms rnms");
}