#include "pragmaHandlerHLS.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "mlir/IR/Builders.h"

using namespace clang;

std::vector<HLSPragmaMetaData> g_HLSMetadata;

PragmaHLSHandler::PragmaHLSHandler(std::vector<HLSPragmaMetaData> &Metadata)
    : PragmaHandler("HLS"), HLSMetaData(Metadata) {}
  
void PragmaHLSHandler::HandlePragma(Preprocessor &PP,
                                 PragmaIntroducer Introducer,
                                 Token &PragmaTok) {
  SourceManager &SM = PP.getSourceManager();
  Token Tok{};
  PP.Lex(Tok);

  if (Tok.isNot(tok::identifier)) return;
  HLSPragmaMetaData P;
  P.Name = StringRef(PP.getSpelling(Tok)).lower();
  P.Loc = Tok.getLocation();
  P.OffsetLoc = SM.getSpellingLineNumber(P.Loc);   // ← add this line
  PP.Lex(Tok);

  if (Tok.is(tok::eod)) {
    HLSMetaData.push_back(P);
    return;
  }

  SourceLocation Start = Tok.getLocation();
  SourceLocation End = Start;
  while (PP.Lex(Tok), Tok.isNot(tok::eod))
    End = Tok.getEndLoc();

  // Extract raw text from source
  bool Invalid = false;
  const char *StartPtr = SM.getCharacterData(Start, &Invalid);
  const char *EndPtr   = SM.getCharacterData(End,   &Invalid);

  if (!Invalid && StartPtr && EndPtr && EndPtr > StartPtr) {
    StringRef RawPragma(StartPtr, EndPtr - StartPtr);
    P.Args = RawPragma.str();
  }
  llvm::outs() << "[HLS] pragma name: " << P.Name << " pragma args: "
               << P.Args << "\n";
  HLSMetaData.push_back(P);
}

void addPragmaHLSHandlers(Preprocessor &PP) {
  PP.AddPragmaHandler(new PragmaHLSHandler(g_HLSMetadata));
}