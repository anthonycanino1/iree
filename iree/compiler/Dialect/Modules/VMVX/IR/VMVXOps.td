// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_VMVX_OPS
#define IREE_DIALECT_MODULES_VMVX_OPS

include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VMVX_PureOp<string mnemonic, list<OpTrait> traits = []> :
    VMVX_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

def VMVX_AddI32Op : VMVX_PureOp<"add.i32", [Commutative]> {
  let arguments = (ins
    I32:$lhs,
    I32:$rhs
  );

  let results = (outs
    I32:$result
  );

  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def VMVX_AddI64Op : VMVX_PureOp<"add.i64", [Commutative]> {
  let arguments = (ins
    I64:$lhs,
    I64:$rhs
  );

  let results = (outs
    I64:$result
  );

  let assemblyFormat = "operands attr-dict `:` type($result)";
}


//===----------------------------------------------------------------------===//
// VMVX Ops: ABI
//===----------------------------------------------------------------------===//

#endif  // IREE_DIALECT_MODULES_VMVX_OPS
