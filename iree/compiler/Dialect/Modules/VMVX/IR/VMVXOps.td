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

def VMVX_AddSI32Op : 
    VMVX_Op<"add.si32", [
      MemoryEffects<[MemRead,MemWrite]>
    ]> {
  let arguments = (ins
    VMVX_Buffer:$lhs_buffer,
    VMVX_Index:$lhs_offset,
    VMVX_Buffer:$rhs_buffer,
    VMVX_Index:$rhs_offset,
    VMVX_Buffer:$dst_buffer,
    VMVX_Index:$dst_offset,
    VMVX_Index:$length
  );

  let assemblyFormat = [{
    `lhs` `(` $lhs_buffer `:` type($lhs_buffer) `)` `[` $lhs_offset `]`
    `rhs` `(` $rhs_buffer `:` type($rhs_buffer) `)` `[` $rhs_offset `]`
    `dst` `(` $dst_buffer `:` type($dst_buffer) `)` `[` $dst_offset `]`
    `length` `(` $length `)` attr-dict
  }];
}

def VMVX_AddSI64Op : 
    VMVX_Op<"add.si64", [
      MemoryEffects<[MemRead,MemWrite]>
    ]> {
  let arguments = (ins
    VMVX_Buffer:$lhs_buffer,
    VMVX_Index:$lhs_offset,
    VMVX_Buffer:$rhs_buffer,
    VMVX_Index:$rhs_offset,
    VMVX_Buffer:$dst_buffer,
    VMVX_Index:$dst_offset,
    VMVX_Index:$length
  );

  let assemblyFormat = [{
    `lhs` `(` $lhs_buffer `:` type($lhs_buffer) `)` `[` $lhs_offset `]`
    `rhs` `(` $rhs_buffer `:` type($rhs_buffer) `)` `[` $rhs_offset `]`
    `dst` `(` $dst_buffer `:` type($dst_buffer) `)` `[` $dst_offset `]`
    `length` `(` $length `)` attr-dict
  }];
}


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
