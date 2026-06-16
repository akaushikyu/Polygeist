#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"  // for SplitString
#include "pragmaHandlerHLS.h"
#include "HandleHLS.h"

using mlir::memref::AllocaOp;
using namespace llvm;

unsigned getSourceInfo(mlir::Location loc) {
  // 1. Base Case: Standard File/Line/Column
  if (auto fileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(loc)) {
    return fileLoc.getLine();
  } 
  if (auto fusedLoc = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
    auto locations = fusedLoc.getLocations();
    if(locations.size() > 0) {
      return getSourceInfo(locations[0]);
    }
    return 0;
  } 
  // 3. Recursive Case: CallSite Locations (If inlining has occurred)
  if (auto callLoc = llvm::dyn_cast<mlir::CallSiteLoc>(loc)) {
    return getSourceInfo(callLoc.getCaller());
  } 
  // 4. Case: Name Location (e.g., loc("myVar"(...) )
  if (auto nameLoc = llvm::dyn_cast<mlir::NameLoc>(loc)) {
    return getSourceInfo(nameLoc.getChildLoc());
  }
  return 0;
} 

void ResolveScope(mlir::ModuleOp MlirModule,
                  std::vector<HLSPragmaMetaData> &HLSMetadata) {

  struct ScopeInfo {
    mlir::Operation *Op;
    unsigned StartLine;
    unsigned EndLine;
  };

  std::vector<ScopeInfo> Scopes;

  // --- Phase 1: Collect all scope ops and their line ranges ---
  MlirModule.walk([&](mlir::Operation *Op) {

    if (llvm::isa<mlir::affine::AffineForOp,
                         mlir::scf::ForOp,
                         mlir::scf::WhileOp,
                         mlir::func::FuncOp>(Op)) {
          llvm::outs() << "[scope] op=" << Op->getName().getStringRef();
    }

    if (mlir::isa<mlir::ModuleOp>(Op))
      return;

    // Only ops that define a new scope (have regions)
    if (Op->getNumRegions() == 0)
      return;

    

    unsigned StartLine = getSourceInfo(Op->getLoc());
    if (StartLine == 0)
      return;
    // End line = max line of any nested op in this scope
    unsigned EndLine = StartLine;
    Op->walk([&](mlir::Operation *Nested) {
      if (Nested == Op)
        return;
      unsigned Line = getSourceInfo(Nested->getLoc());
      if (Line > EndLine)
        EndLine = Line;
    });
    Scopes.push_back({Op, StartLine, EndLine});
    llvm::outs() << "[scope] op=" << Op->getName().getStringRef()
           << " start=" << StartLine << " end=" << EndLine << "\n";
  });


  // --- Phase 1.5: Widen each func.func's StartLine to cover the gap from
  // the previous function's end. This lets pragmas that appear above the
  // first emitted op (between '{' and the first statement) — and pragmas
  // between functions — attach to the function that follows them.
  // Other scope ops (loops) keep their original tight range, so the
  // innermost-loop preference still works.
  std::vector<ScopeInfo *> Funcs;
  for (auto &S : Scopes)
    if (llvm::isa<mlir::func::FuncOp>(S.Op))
      Funcs.push_back(&S);

  llvm::sort(Funcs, [](ScopeInfo *a, ScopeInfo *b) {
    return a->EndLine < b->EndLine;
  });

  unsigned PrevEnd = 0;
  for (auto *F : Funcs) {
    F->StartLine = PrevEnd + 1;
    PrevEnd = F->EndLine;
  }

  // --- Phase 2: Match each pragma to its innermost containing scope ---
  for (HLSPragmaMetaData &Meta : HLSMetadata) {
    mlir::Operation *InnermostOp   = nullptr;
    unsigned         InnermostSpan = UINT_MAX;

    for (auto &Scope : Scopes) {
      // Pragma must fall within the scope's line range
      if (Meta.OffsetLoc < Scope.StartLine || Meta.OffsetLoc > Scope.EndLine)
        continue;

      // Prefer the tightest (innermost) enclosing scope
      unsigned Span = Scope.EndLine - Scope.StartLine;
      // After --raise-scf-to-affine, loops are affine.for; before that, scf.for/while.
      bool CurrIsLoop = llvm::isa<mlir::affine::AffineForOp,
                                  mlir::scf::ForOp,
                                  mlir::scf::WhileOp>(Scope.Op);
      bool BestIsLoop = InnermostOp &&
                        llvm::isa<mlir::affine::AffineForOp,
                                  mlir::scf::ForOp,
                                  mlir::scf::WhileOp>(InnermostOp);

      if (Span < InnermostSpan || (CurrIsLoop && !BestIsLoop && Span == InnermostSpan)) {
        InnermostSpan = Span;
        InnermostOp   = Scope.Op;
      }
    }

    if (InnermostOp) {
      Meta.Op = InnermostOp;
      llvm::outs() << "[HLS] pragma '" << Meta.Name
             << "' at line " << Meta.OffsetLoc
             << " attached to '" << InnermostOp->getName().getStringRef()
             << "' (scope span " << InnermostSpan << " lines)";

      if (auto SymAttr = InnermostOp->getAttrOfType<mlir::StringAttr>(
              mlir::SymbolTable::getSymbolAttrName()))

        llvm::outs() << " ['" << SymAttr.getValue() << "']";
      llvm::outs() << "\n";
    } else {
      Meta.Op = nullptr;
      llvm::outs() << "[HLS] Warning: pragma '" << Meta.Name
                   << "' at line " << Meta.OffsetLoc
                   << " could not be resolved to any scope\n";
    }
  }
}

// Helper: parse a key=value arg list into a StringMap
void ParseArgs(ArrayRef<StringRef> Args,
                      llvm::StringMap<StringRef> &KV,
                      SmallVectorImpl<StringRef> &Flags) {
  for (StringRef Arg : Args) {
    if (Arg.contains('=')) {
      auto [Key, Val] = Arg.split('=');
      KV[Key] = Val;
    } else {
      Flags.push_back(Arg); // bare keywords like "complete", "rewind", "skip_exit_check"
    }
  }
}

// Helper: find a memref.alloca by base variable name inside a scope op.
// memref.alloca has no name attribute; we recover the variable name from
// either (a) a NameLoc on the op's location, or (b) a "polygeist.varname"
// string attribute attached at scan time. Either source works; the second
// is the fallback if locations have been rewritten by passes.
AllocaOp FindAlloca(mlir::Operation *ScopeOp, StringRef BaseName) {
  AllocaOp Found;
  ScopeOp->walk([&](AllocaOp Alloca) {
    // (a) try NameLoc on the alloca's location
    if (auto NL = llvm::dyn_cast<mlir::NameLoc>(Alloca.getLoc())) {
      if (NL.getName().getValue() == BaseName) {
        Found = Alloca;
        return;
      }
    }
    // (b) try a polygeist.varname attribute
    if (auto NameAttr =
            Alloca->getAttrOfType<mlir::StringAttr>("polygeist.varname")) {
      if (NameAttr.getValue() == BaseName) {
        Found = Alloca;
        return;
      }
    }
  });
  return Found;
}


mlir::Operation *FindOpByName(mlir::Operation *ScopeOp, llvm::StringRef Name) {
  mlir::Operation *Found = nullptr;
  ScopeOp->walk([&](mlir::Operation *Op) {
    if (auto names = Op->getAttrOfType<mlir::ArrayAttr>("polygeist.ssa_names")) {
      for (auto attr : names) {
        if (auto sa = llvm::dyn_cast<mlir::StringAttr>(attr)) {
          if (sa.getValue() == Name) {
            Found = Op;
            return mlir::WalkResult::interrupt();
          }
        }
      }
    }
    return mlir::WalkResult::advance();
  });
  return Found;
}

// Helper: append a DictionaryAttr entry to an ArrayAttr on an op
void AppendAttrEntry(mlir::Operation *Op, StringRef AttrName,
                            mlir::DictionaryAttr Entry) {
  mlir::Builder B(Op->getContext());
  SmallVector<mlir::Attribute, 4> Entries;
  if (auto Existing = Op->getAttrOfType<mlir::ArrayAttr>(AttrName))
    Entries.append(Existing.begin(), Existing.end());
  Entries.push_back(Entry);
  Op->setAttr(AttrName, B.getArrayAttr(Entries));
}

// Helper: parse integer from StringRef, returns -1 on failure
int ParseInt(StringRef S) {
  int Val = -1;
  S.getAsInteger(10, Val);
  return Val;
}

// ── array_partition ────────────────────────────────────────────────────────
void HandleArrayPartition(HLSPragmaMetaData &Meta) {
  SmallVector<StringRef, 8> Args;
  SplitString(Meta.Args, Args);

  llvm::StringMap<StringRef> KV;
  SmallVector<StringRef, 4> Flags;
  ParseArgs(Args, KV, Flags);

  StringRef VarPath = KV.lookup("variable");
  if (VarPath.empty()) {
    llvm::outs() << "[HLS] Warning: array_partition missing variable=\n";
    return;
  }

  StringRef Base = VarPath.split('.').first;
  AllocaOp AttachOp = FindAlloca(Meta.Op, Base);
  if (!AttachOp) {
    llvm::outs() << "[HLS] Warning: array_partition could not find alloca '"
                 << Base << "'\n";
    return;
  }

  mlir::Builder B(AttachOp.getContext());
  SmallVector<mlir::NamedAttribute, 5> Fields;
  Fields.push_back(B.getNamedAttr("variable", B.getStringAttr(VarPath)));

  // kind: explicit key or bare flag (complete/block/cyclic), default complete
  StringRef Kind = KV.count("type") ? KV["type"] : StringRef("complete");
  for (StringRef F : Flags)
    if (F == "complete" || F == "block" || F == "cyclic") { Kind = F; break; }
  Fields.push_back(B.getNamedAttr("kind", B.getStringAttr(Kind)));

  if (KV.count("factor"))
    Fields.push_back(B.getNamedAttr("factor",
        B.getI32IntegerAttr(ParseInt(KV["factor"]))));
  if (KV.count("dim"))
    Fields.push_back(B.getNamedAttr("dim",
        B.getI32IntegerAttr(ParseInt(KV["dim"]))));

  AppendAttrEntry(AttachOp, "hls.array_partition", B.getDictionaryAttr(Fields));
  llvm::outs() << "[HLS] Attached array_partition {variable=" << VarPath
               << ", kind=" << Kind << "} to '" << Base << "'\n";
}

// ── array_reshape ──────────────────────────────────────────────────────────
// Same args as array_partition; semantics differ (word-width widening vs split)
void HandleArrayReshape(HLSPragmaMetaData &Meta) {
  SmallVector<StringRef, 8> Args;
  SplitString(Meta.Args, Args);

  llvm::StringMap<StringRef> KV;
  SmallVector<StringRef, 4> Flags;
  ParseArgs(Args, KV, Flags);

  StringRef VarPath = KV.lookup("variable");
  if (VarPath.empty()) {
    llvm::outs() << "[HLS] Warning: array_reshape missing variable=\n";
    return;
  }

  StringRef Base = VarPath.split('.').first;
  AllocaOp AttachOp = FindAlloca(Meta.Op, Base);
  if (!AttachOp) {
    llvm::outs() << "[HLS] Warning: array_reshape could not find alloca '"
                 << Base << "'\n";
    return;
  }

  mlir::Builder B(AttachOp.getContext());
  SmallVector<mlir::NamedAttribute, 5> Fields;
  Fields.push_back(B.getNamedAttr("variable", B.getStringAttr(VarPath)));

  StringRef Kind = KV.count("type") ? KV["type"] : StringRef("complete");
  for (StringRef F : Flags)
    if (F == "complete" || F == "block" || F == "cyclic") { Kind = F; break; }
  Fields.push_back(B.getNamedAttr("kind", B.getStringAttr(Kind)));

  if (KV.count("factor"))
    Fields.push_back(B.getNamedAttr("factor",
        B.getI32IntegerAttr(ParseInt(KV["factor"]))));
  if (KV.count("dim"))
    Fields.push_back(B.getNamedAttr("dim",
        B.getI32IntegerAttr(ParseInt(KV["dim"]))));

  AppendAttrEntry(AttachOp, "hls.array_reshape", B.getDictionaryAttr(Fields));
  llvm::outs() << "[HLS] Attached array_reshape {variable=" << VarPath
               << ", kind=" << Kind << "} to '" << Base << "'\n";
}

// ── bind_storage ───────────────────────────────────────────────────────────
// #pragma HLS bind_storage variable=<v> type=<ram_1p|ram_2p|fifo|...>
//                           impl=<bram|uram|lutram|...> latency=<int>
void HandleBindStorage(HLSPragmaMetaData &Meta) {
  SmallVector<StringRef, 8> Args;
  SplitString(Meta.Args, Args);

  llvm::StringMap<StringRef> KV;
  SmallVector<StringRef, 4> Flags;
  ParseArgs(Args, KV, Flags);

  StringRef VarPath = KV.lookup("variable");
  if (VarPath.empty()) {
    llvm::outs() << "[HLS] Warning: bind_storage missing variable=\n";
    return;
  }

  StringRef Base = VarPath.split('.').first;
  AllocaOp AttachOp = FindAlloca(Meta.Op, Base);
  if (!AttachOp) {
    llvm::outs() << "[HLS] Warning: bind_storage could not find alloca '"
                 << Base << "'\n";
    return;
  }

  mlir::Builder B(AttachOp.getContext());
  SmallVector<mlir::NamedAttribute, 5> Fields;
  Fields.push_back(B.getNamedAttr("variable", B.getStringAttr(VarPath)));

  if (KV.count("type"))
    Fields.push_back(B.getNamedAttr("type", B.getStringAttr(KV["type"])));
  if (KV.count("impl"))
    Fields.push_back(B.getNamedAttr("impl", B.getStringAttr(KV["impl"])));
  if (KV.count("latency"))
    Fields.push_back(B.getNamedAttr("latency",
        B.getI32IntegerAttr(ParseInt(KV["latency"]))));

  AppendAttrEntry(AttachOp, "hls.bind_storage", B.getDictionaryAttr(Fields));
  llvm::outs() << "[HLS] Attached bind_storage {variable=" << VarPath << "} to '"
               << Base << "'\n";
}

// ── bind_op ────────────────────────────────────────────────────────────────
// #pragma HLS bind_op variable=<v> op=<add|mul|...>
//                     impl=<dsp|fabric> latency=<int>
void HandleBindOp(HLSPragmaMetaData &Meta) {
  SmallVector<StringRef, 8> Args;
  SplitString(Meta.Args, Args);

  llvm::StringMap<StringRef> KV;
  SmallVector<StringRef, 4> Flags;
  ParseArgs(Args, KV, Flags);

  StringRef VarPath = KV.lookup("variable");
  if (VarPath.empty()) {
    llvm::outs() << "[HLS] Warning: bind_op missing variable=\n";
    return;
  }

  StringRef Base = VarPath.split('.').first;
  mlir::Operation *AttachOp = FindOpByName(Meta.Op, Base);

  if (!AttachOp) {
    llvm::outs() << "[HLS] Warning: bind_op could not find producer for '"
                 << Base << "'\n";
    return;
  }

  // Optionally: filter by op type matching the pragma's op= field
  if (KV.count("op")) {
    StringRef Want = KV["op"];
    bool ok = false;
    if (Want == "mul")
      ok = llvm::isa<mlir::arith::MulIOp, mlir::arith::MulFOp>(AttachOp);
    else if (Want == "add")
      ok = llvm::isa<mlir::arith::AddIOp, mlir::arith::AddFOp>(AttachOp);
    // ... etc
    if (!ok) {
      llvm::outs() << "[HLS] Warning: bind_op variable=" << Base
                   << " op=" << Want << " but producer is "
                   << AttachOp->getName().getStringRef() << "\n";
      return;
    }
  }

  mlir::Builder B(AttachOp->getContext());
  SmallVector<mlir::NamedAttribute, 4> Fields;
  if (KV.count("op"))      Fields.push_back(B.getNamedAttr("op",      B.getStringAttr(KV["op"])));
  if (KV.count("impl"))    Fields.push_back(B.getNamedAttr("impl",    B.getStringAttr(KV["impl"])));
  if (KV.count("latency")) Fields.push_back(B.getNamedAttr("latency", B.getI32IntegerAttr(ParseInt(KV["latency"]))));

  AppendAttrEntry(AttachOp, "hls.bind_op", B.getDictionaryAttr(Fields));
  llvm::outs() << "[HLS] Attached bind_op {variable=" << VarPath << "} to '"
               << Base << "'\n";
}

// ── allocation ─────────────────────────────────────────────────────────────
// #pragma HLS allocation instances=<func_name> limit=<int> type=<function|operation>
// No variable — attaches to the enclosing function/scope op directly
void HandleAllocation(HLSPragmaMetaData &Meta) {
  SmallVector<StringRef, 8> Args;
  SplitString(Meta.Args, Args);

  llvm::StringMap<StringRef> KV;
  SmallVector<StringRef, 4> Flags;
  ParseArgs(Args, KV, Flags);

  if (!KV.count("instances") || !KV.count("limit")) {
    llvm::outs() << "[HLS] Warning: allocation missing instances= or limit=\n";
    return;
  }

  mlir::Builder B(Meta.Op->getContext());
  SmallVector<mlir::NamedAttribute, 4> Fields;
  Fields.push_back(B.getNamedAttr("instances",
      B.getStringAttr(KV["instances"])));
  Fields.push_back(B.getNamedAttr("limit",
      B.getI32IntegerAttr(ParseInt(KV["limit"]))));

  // type= defaults to "function"
  StringRef AllocType = KV.count("type") ? KV["type"] : StringRef("function");
  Fields.push_back(B.getNamedAttr("type", B.getStringAttr(AllocType)));

  AppendAttrEntry(Meta.Op, "hls.allocation", B.getDictionaryAttr(Fields));
  llvm::outs() << "[HLS] Attached allocation {instances=" << KV["instances"]
               << ", limit=" << KV["limit"] << "} to scope op\n";
}

// ── unroll ─────────────────────────────────────────────────────────────────
// #pragma HLS unroll [factor=<int>] [skip_exit_check]
// No variable — attaches to the enclosing loop/scope op
void HandleUnroll(HLSPragmaMetaData &Meta) {
  SmallVector<StringRef, 8> Args;
  SplitString(Meta.Args, Args);

  llvm::StringMap<StringRef> KV;
  SmallVector<StringRef, 4> Flags;
  ParseArgs(Args, KV, Flags);

  mlir::Builder B(Meta.Op->getContext());
  SmallVector<mlir::NamedAttribute, 3> Fields;

  // factor= absent means full unroll
  if (KV.count("factor"))
    Fields.push_back(B.getNamedAttr("factor",
        B.getI32IntegerAttr(ParseInt(KV["factor"]))));
  else
    Fields.push_back(B.getNamedAttr("factor", B.getI32IntegerAttr(0))); // 0 = full

  bool SkipExitCheck = llvm::is_contained(Flags, StringRef("skip_exit_check"));
  Fields.push_back(B.getNamedAttr("skip_exit_check",
      B.getBoolAttr(SkipExitCheck)));

  // Attach directly to the scope op (loop body)
  AppendAttrEntry(Meta.Op, "hls.unroll", B.getDictionaryAttr(Fields));
  llvm::outs() << "[HLS] Attached unroll {factor="
               << (KV.count("factor") ? KV["factor"] : StringRef("full"))
               << ", skip_exit_check=" << SkipExitCheck << "} to scope op\n";
}

// ── pipeline ───────────────────────────────────────────────────────────────
// #pragma HLS pipeline [II=<int>] [rewind] [style=stp|flp|frp]
// No variable — attaches to the enclosing function/loop scope op
void HandlePipeline(HLSPragmaMetaData &Meta) {
  SmallVector<StringRef, 8> Args;
  SplitString(Meta.Args, Args);

  llvm::StringMap<StringRef> KV;
  SmallVector<StringRef, 4> Flags;
  ParseArgs(Args, KV, Flags);

  mlir::Builder B(Meta.Op->getContext());
  SmallVector<mlir::NamedAttribute, 4> Fields;

  // II=0 means "let tool decide minimum"
  int II = KV.count("II") ? ParseInt(KV["II"]) : 0;
  Fields.push_back(B.getNamedAttr("II", B.getI32IntegerAttr(II)));

  bool Rewind = llvm::is_contained(Flags, StringRef("rewind"));
  Fields.push_back(B.getNamedAttr("rewind", B.getBoolAttr(Rewind)));

  // style: stp (stall), flp (flushable), frp (free-running), default stp
  StringRef Style = KV.count("style") ? KV["style"] : StringRef("stp");
  Fields.push_back(B.getNamedAttr("style", B.getStringAttr(Style)));

  // Attach directly to the scope op (function or loop)
  AppendAttrEntry(Meta.Op, "hls.pipeline", B.getDictionaryAttr(Fields));
  llvm::outs() << "[HLS] Attached pipeline {II=" << II
               << ", rewind=" << Rewind << ", style=" << Style
               << "} to scope op\n";
}



void AttachMetadata(std::vector<HLSPragmaMetaData> &HLSMetadata) {
  for (HLSPragmaMetaData &Meta : HLSMetadata) {
    llvm::outs() << "Arguments: " << Meta.Args << "\n";
    if (!Meta.Op) {
      llvm::outs() << "[HLS] Warning: pragma '" << Meta.Name
                   << "' has no resolved scope, skipping\n";
      continue;
    }
    if (Meta.Name == "array_partition") HandleArrayPartition(Meta);
    else if (Meta.Name == "array_reshape")   HandleArrayReshape(Meta);
    else if (Meta.Name == "bind_storage")    HandleBindStorage(Meta);
    else if (Meta.Name == "bind_op")         HandleBindOp(Meta);
    else if (Meta.Name == "allocation")      HandleAllocation(Meta);
    else if (Meta.Name == "unroll")          HandleUnroll(Meta);
    else if (Meta.Name == "pipeline")        HandlePipeline(Meta);
  }
}

