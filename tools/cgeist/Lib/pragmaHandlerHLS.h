#ifndef MLIR_TOOLS_MLIRCLANG_LIB_PRAGMAHANDLERHLS_H
#define MLIR_TOOLS_MLIRCLANG_LIB_PRAGMAHANDLERHLS_H

#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/Pragma.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "mlir/IR/Operation.h"
#include <string>
#include <vector>

enum HLSPragmaType {
    ArrayPartition,
    ArrayReshape,
    BindStorage,
    BindOp,
    Allocation,
    Unroll,
    Pipeline
  };

/// Holds metadata parsed from #pragma HLS
struct HLSPragmaMetaData {
    std::string Name;      // e.g., "pipeline"
    std::string Args;      // e.g., "II=1"
    clang::SourceLocation Loc;
    mlir::Operation *Op = nullptr;
    unsigned OffsetLoc = 0;
};

/// Handles the transition from Preprocessor (#pragma) to AST metadata
class PragmaHLSHandler : public clang::PragmaHandler {
public:
    PragmaHLSHandler(std::vector<HLSPragmaMetaData> &Metadata);
        
    void HandlePragma(clang::Preprocessor &PP, 
                      clang::PragmaIntroducer Introducer, 
                      clang::Token &PragmaTok) override;
private:
    std::vector<HLSPragmaMetaData> &HLSMetaData;
};

void addPragmaHLSHandlers(clang::Preprocessor &PP);

// Global storage for HLS pragma metadata collected during parse.
// Populated by PragmaHLSHandler::HandlePragma, consumed in driver.cc
// after the MLIR pass pipeline runs.
extern std::vector<HLSPragmaMetaData> g_HLSMetadata;


#endif