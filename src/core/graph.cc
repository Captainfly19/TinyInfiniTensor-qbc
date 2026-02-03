#include "core/graph.h"
#include "core/op_type.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>    
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
 
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================


  bool optimized = true;
  while (optimized) {
    optimized = false;
    for (auto op :ops) {
      if (op->type == OpType::Transpose) {

        for(auto next_op :op->getSuccessors()){
            if (next_op->type == OpType::Transpose) {
            
            auto perm1 = as<TransposeObj>(op)->getPermute();
            auto perm2 = as<TransposeObj>(next_op)->getPermute();

            if (perm1 == perm2) {
                for(auto last_op : op->getPredecessors()){
                    for(auto output :next_op->getOutputs()){
                        output->setSource(last_op);
                    }
                }                            

                for(auto input :op->getInputs()){
                    for(auto nex_op :next_op->getSuccessors()){
                        input->removeTarget(op);
                        input->addTarget(nex_op);
                        nex_op->removePredecessors(next_op);
                        nex_op->replaceInput(next_op->getOutput(),input); 
                    }
                }     

                for(auto last_op :op->getPredecessors()){
                    last_op->removeSuccessors(op);
                    for(auto nex_op :next_op->getSuccessors()){
                        last_op->addSuccessors(nex_op);
                        nex_op->addPredecessors(last_op);
                    }
                }


                removeTensor(next_op->getOutput());
                removeTensor(op->getOutput());
                removeOperator(next_op);
                removeOperator(op);
            } 


            optimized = true;
            break;
            }
        }
    }

      if (op->type == OpType::MatMul) {
        for (auto input : op->getInputs()) {
          if (input->getSource() != nullptr && input->getSource()->type == OpType::Transpose){
            auto preop = input->getSource();
            auto transpose_op = dynamic_cast<TransposeObj*>(preop.get());

            auto perm = transpose_op->getPermute();
            if (std::all_of(perm.begin(),perm.begin()+perm.size()-2,[ i =0](int p )mutable {return p==i++;}))
                {

                    if (perm[perm.size() - 1] - perm[perm.size() - 2] == -1){
                            auto matop = dynamic_cast<MatmulObj*>(op.get());

                            if (input == op->getInputs()[0]) {
                                matop->setTransA(!matop->getTransA());
                            } else {
                                matop->setTransB(!matop->getTransB());
                            }

                            for(auto transinput :transpose_op->getInputs() ){
                                transinput->removeTarget(preop);
                                transinput->addTarget(op);
                                op->replaceInput(input,transinput);
                            }
                            op->removePredecessors(preop);
                            for(auto pred :transpose_op->getPredecessors()){
                                pred->removeSuccessors(preop);
                                pred->addSuccessors(op);
                                op->addPredecessors(pred);
                            }

                            removeTensor(transpose_op->getOutputs()[0]);
                            removeOperator(preop);

                            optimized = true;
                    }
                }
            }
        }
        if(optimized) break;

      }
    }
  
  }
}

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // 用新的输出形状和大小代替旧的
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc() {
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业
        // ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给
        // tensor 绑定内存
        // =================================== 作业
        // ===================================
        size_t total_size{0};
        for(auto &tensor :tensors){
            total_size += tensor->getBytes();
        }
        size_t addr = allocator.alloc((total_size));
        size_t offset = 0;
        for(auto &tensor :tensors){
            tensor->setDataBlob(make_ref<BlobObj>(runtime,reinterpret_cast<void*>(reinterpret_cast<char*>(allocator.getPtr()) +addr+ offset)));
            offset += tensor->getBytes();
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // 检查是否存在两个具有相同FUID的张量
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini