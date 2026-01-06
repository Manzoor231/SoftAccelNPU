#pragma once

#include "softaccelnpu/tensor.h"
#include "softaccelnpu/device_manager.h"
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace softaccelnpu {

// Forward declarations
class DmlCommandList;
class DmlOperator;

/**
 * @brief Represents a logical NPU device, similar to IDMLDevice.
 */
class DmlDevice {
public:
    static std::shared_ptr<DmlDevice> create();
    
    std::shared_ptr<DmlCommandList> create_command_list();
    std::shared_ptr<DmlOperator> create_gemm_operator(size_t M, size_t N, size_t K);
    
    // Performance stats
    void print_report();
};

/**
 * @brief Descriptor for Gemm operation.
 */
struct DmlGemmDescriptor {
    size_t M, N, K;
    float alpha = 1.0f;
    float beta = 0.0f;
};

/**
 * @brief Represents a compiled operator, similar to IDMLCompiledOperator.
 */
class DmlOperator {
public:
    enum class Ty { GEMM, ELEMENTWISE_BIAS, ACTIVATION };
    enum class ActivationTy { RELU, SILU };
    
    DmlOperator(Ty type, DmlGemmDescriptor desc) : type_(type), gemm_desc_(desc) {}
    DmlOperator(Ty type, ActivationTy act) : type_(type), activation_ty_(act) {}
    
    Ty get_type() const { return type_; }
    const DmlGemmDescriptor& get_gemm_desc() const { return gemm_desc_; }
    ActivationTy get_activation_type() const { return activation_ty_; }

private:
    Ty type_;
    DmlGemmDescriptor gemm_desc_;
    ActivationTy activation_ty_ = ActivationTy::RELU;
};

/**
 * @brief Records operations for execution, similar to IDMLCommandRecorder/List.
 */
class DmlCommandList {
public:
    void record_gemm(
        std::shared_ptr<DmlOperator> op,
        const Tensor& A,
        const Tensor& B,
        Tensor& C
    );

    void record_bias_add(
        const Tensor& input,
        const Tensor& bias,
        Tensor& output
    );

    void record_activation(
        std::shared_ptr<DmlOperator> op,
        const Tensor& input,
        Tensor& output
    );

    /**
     * @brief Executes all recorded commands.
     */
    void execute();

    void reset();

private:
    struct Command {
        DmlOperator::Ty type;
        std::shared_ptr<DmlOperator> op;
        const Tensor* A; // or input
        const Tensor* B; // or bias
        Tensor* C;       // or output
    };
    std::vector<Command> commands_;
};

/**
 * @brief Simplified Binding Table for mapping resources.
 */
class DmlBindingTable {
    // In this simplified implementation, we bind directly in the record call.
    // This class is kept for API compatibility/future-proofing.
};

} // namespace softaccelnpu
