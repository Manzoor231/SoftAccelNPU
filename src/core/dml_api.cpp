#include "softaccelnpu/dml_api.h"
#include "softaccelnpu/ops.h"
#include "softaccelnpu/npu_driver.h"
#include "softaccelnpu/cache_model.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace softaccelnpu {

// --- DmlDevice ---

std::shared_ptr<DmlDevice> DmlDevice::create() {
    return std::make_shared<DmlDevice>();
}

std::shared_ptr<DmlCommandList> DmlDevice::create_command_list() {
    return std::make_shared<DmlCommandList>();
}

std::shared_ptr<DmlOperator> DmlDevice::create_gemm_operator(size_t M, size_t N, size_t K) {
    DmlGemmDescriptor desc;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    return std::make_shared<DmlOperator>(DmlOperator::Ty::GEMM, desc);
}

void DmlDevice::print_report() {
    CacheModel::print_4d_report();
}

// --- DmlCommandList ---

void DmlCommandList::record_gemm(
    std::shared_ptr<DmlOperator> op,
    const Tensor& A,
    const Tensor& B,
    Tensor& C
) {
    commands_.push_back({DmlOperator::Ty::GEMM, op, &A, &B, &C});
}

void DmlCommandList::record_bias_add(
    const Tensor& input,
    const Tensor& bias,
    Tensor& output
) {
    commands_.push_back({DmlOperator::Ty::ELEMENTWISE_BIAS, nullptr, &input, &bias, &output});
}

void DmlCommandList::record_activation(
    std::shared_ptr<DmlOperator> op,
    const Tensor& input,
    Tensor& output
) {
    commands_.push_back({DmlOperator::Ty::ACTIVATION, op, &input, nullptr, &output});
}

void DmlCommandList::execute() {
    for (const auto& cmd : commands_) {
        if (cmd.type == DmlOperator::Ty::GEMM) {
            GemmOps::gemm_tiled(*cmd.A, *cmd.B, *cmd.C);
        } else if (cmd.type == DmlOperator::Ty::ELEMENTWISE_BIAS) {
            // Simple bias add
            float* in = (float*)cmd.A->data();
            float* b = (float*)cmd.B->data();
            float* out = (float*)cmd.C->data();
            for (size_t i = 0; i < cmd.A->rows() * cmd.A->cols(); ++i) {
                out[i] = in[i] + b[i % cmd.B->rows()];
            }
        } else if (cmd.type == DmlOperator::Ty::ACTIVATION) {
            float* in = (float*)cmd.A->data();
            float* out = (float*)cmd.C->data();
            auto act = cmd.op->get_activation_type();
            for (size_t i = 0; i < cmd.A->rows() * cmd.A->cols(); ++i) {
                if (act == DmlOperator::ActivationTy::RELU) {
                    out[i] = std::max(0.0f, in[i]);
                } else if (act == DmlOperator::ActivationTy::SILU) {
                    out[i] = in[i] / (1.0f + std::exp(-in[i]));
                }
            }
        }
    }
}

void DmlCommandList::reset() {
    commands_.clear();
}

} // namespace softaccelnpu
