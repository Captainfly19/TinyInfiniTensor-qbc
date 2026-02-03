#include "operators/matmul.h"
#include <utility>

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
		// 最后两个维度矩阵乘 前面的应该是广播即可
		const auto A = inputs[0];
		const auto B = inputs[1];
		auto A_dim = A->getDims();
		auto B_dim = B->getDims();

		auto A_rank = A->getRank();
		auto B_rank = B->getRank();

		if(transA){
			std::swap(A_dim[A_rank-1],A_dim[A_rank-2]);
		}
		if(transB){
			std::swap(B_dim[B_rank-1],B_dim[B_rank-2]);
		}

		// if(A_dim[A_rank-1] != B_dim[B_rank-2]){
		// 	return std::nullopt;
		// }

		size_t max_size = std::max(A_rank, B_rank);
		Shape output_dim(max_size);

		output_dim[max_size-1] = B_dim[B_rank-1];
		output_dim[max_size-2] = A_dim[B_rank-2];

		for(size_t i = 0; i < max_size-2; i++){
			output_dim[i] = std::max(A_dim[i], B_dim[i]);
		}

        return vector<Shape>{output_dim};
    }

} // namespace infini