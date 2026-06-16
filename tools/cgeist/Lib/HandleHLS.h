#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "mlir/IR/Builders.h"

void ResolveScope(mlir::ModuleOp MlirModule,
                  std::vector<HLSPragmaMetaData> &HLSMetadata);

void AttachMetadata(std::vector<HLSPragmaMetaData> &HLSMetadata);
