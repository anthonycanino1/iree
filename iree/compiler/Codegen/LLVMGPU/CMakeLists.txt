################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# iree/compiler/Codegen/LLVMGPU/BUILD                                          #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_cc_library(
  NAME
    LLVMGPU
  HDRS
    "ConvertToLLVM.h"
    "KernelConfig.h"
  SRCS
    "ConvertToLLVM.cpp"
    "ConvertToNVVM.cpp"
    "ConvertToROCDL.cpp"
    "KernelConfig.cpp"
    "LLVMGPULowerExecutableTarget.cpp"
    "LLVMGPURemoveTrivialLoops.cpp"
    "LLVMGPUTileAndDistribute.cpp"
    "LLVMGPUVectorization.cpp"
    "Passes.cpp"
  DEPS
    MLIRAffine
    MLIRAffineToStandard
    MLIRGPU
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRIR
    MLIRLinalg
    MLIRLinalgTransforms
    MLIRMath
    MLIRMemRefTransforms
    MLIRNVVMIR
    MLIRPass
    MLIRROCDLIR
    MLIRSCFToStandard
    MLIRStandard
    MLIRStandardOpsTransforms
    MLIRStandardToLLVM
    MLIRTransforms
    MLIRVector
    MLIRVectorToLLVM
    MLIRVectorToSCF
    iree::compiler::Codegen::Common
    iree::compiler::Codegen::PassHeaders
    iree::compiler::Codegen::Transforms
    iree::compiler::Codegen::Utils
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::LinalgExt::Transforms
    iree::compiler::Dialect::Shape::Transforms
    tensorflow::mlir_hlo
  PUBLIC
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
