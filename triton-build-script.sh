#!/bin/bash

################################################################################
echo " ______    _ __              ___       _ __   __  ____        _      __  "
echo "/_  __/___(_) /____  ___    / _ )__ __(_) /__/ / / __/_______(_)__  / /_ "
echo " / / / __/ / __/ _ \/ _ \  / _  / // / / / _  / _\ \/ __/ __/ / _ \/ __/ "
echo "/_/ /_/ /_/\__/\___/_//_/ /____/\_,_/_/_/\_,_/ /___/\__/_/ /_/ .__/\__/  "
echo "                                                            /_/          "
################################################################################

# How To Run:
# mkdir triton-clean-dir
# cd triton-clean-dir
# bash ./triton-build-script.sh

# Override the exports as you see fit (For CUDA version, torch nightly etc)

# This script will clone a fresh llvm-project and triton with correct pin
# hashes, setup a virtual environment and install all Triton dependencies
# including a nightly PyTorch.

################################################################################

# Before running make sure to run:
# sudo dnf install python3.11-devel python3.11 ccache cmake ninja-build clang llvm lld zlib zlib-devl

################################################################################

function gpu-unified-setup() {
    if hash nvidia-smi 2>/dev/null; then
      echo ""
      echo "Running Nvidia 🟩 Edition..."
      echo ""

      # Set these Version Variables:
      export CUDA_OR_ROCM=cu
      export CUDA_MAJOR=12
      export CUDA_MINOR=6

      # CUDA Path Variables:
      export CUDA_HOME=/usr/local/cuda-$CUDA_MAJOR.$CUDA_MINOR
      export PATH=$CUDA_HOME/bin:$PATH
      export USE_CUDA=1
      export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
      export CUDA_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
    elif hash rocm-smi 2>/dev/null; then
      echo ""
      echo "Running AMD 🟥 Edition..."
      echo ""

      # Set these Version Variables:
      export CUDA_OR_ROCM=rocm
      export CUDA_MAJOR=6
      export CUDA_MINOR=".3"
      # export TRITON_PIN_URL=https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/ci_commit_pins/triton-rocm.txt
    else
      echo "GPU Unsupported for triton-build-script..."
      exit 255
    fi

    export TORCH_URL=https://download.pytorch.org/whl/nightly/$CUDA_OR_ROCM$CUDA_MAJOR$CUDA_MINOR

    echo ""
    echo "🐍🔥 Using Torch Version from $TORCH_URL ..."
    echo ""
}


echo ""
echo "Configuring all the GPU Things..."
echo "Feeding The Fishes 🐟🐠🐡 ..."
echo ""

gpu-unified-setup

# Set These Project Variables
export LLVM_TARGETS="Native;NVPTX;AMDGPU"
export LLVM_PROJECTS="mlir;llvm;"
export LLVM_BUILD_TYPE=Debug
export TRITON_BUILD_DEBUG=1

if [ "$TRITON_SCRIPT_BUILD_FLAVOR" == "RelWithDebInfo" ]; then
  echo "Configuring for Release with Debug Info 🚀🚀🚀 🔨🐞"
  export LLVM_BUILD_TYPE=RelWithDebInfo
  export TRITON_BUILD_DEBUG=0
elif [ "$TRITON_SCRIPT_BUILD_FLAVOR" == "Release" ]; then
  echo "Configuring for Release Mode 🚀🚀🚀"
  export LLVM_BUILD_TYPE=Release
  export TRITON_BUILD_DEBUG=0
else
  echo "Configuring in Debug Mode 🛠️🪲🐞"
fi

export VENV_PROJECT_NAME=triton

# Directory Locations
export PROJECT_DIR=`pwd`
export LLVM_BUILD_DIR=$PROJECT_DIR/llvm-project/build

# Hash Pin URLs
export LLVM_PIN_FILE=$PROJECT_DIR/triton/cmake/llvm-hash.txt
export TRITON_PIN_URL=https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/ci_commit_pins/triton.txt

# Git Repos
export TRITON_REPO=https://github.com/triton-lang/triton
export LLVMPROJECT_REPO=https://github.com/llvm/llvm-project

################################################################################
#
echo ""
echo "Cloning Triton ⚛ and LLVM 🐲 Repos..."
echo ""

# Clone Repos
echo -n "⚛ "
git clone $TRITON_REPO
echo ""
echo ""
echo ""
echo -n "🐲 "
git clone $LLVMPROJECT_REPO

echo ""
echo "Setting Pins..."
echo ""

# Checkout PIN Hashes
echo -n "⚛ "
if [ "$TRITON_SCRIPT_BUILD_VERSION" == "" ]; then
  echo "Using Triton version from Top of Tree main"
elif [ "$TRITON_SCRIPT_BUILD_VERSION" == "main" ]; then
  echo "Using Triton version from Top of Tree main"
elif [ "$TRITON_SCRIPT_BUILD_VERSION" == "pin" ]; then
  echo "Using Triton version based on the PyTorch Pin"
  git -C $PROJECT_DIR/triton reset --hard `curl $TRITON_PIN_URL`
else
  echo "Using Triton version: $TRITON_SCRIPT_BUILD_VERSION"
  git -C $PROJECT_DIR/triton reset --hard $TRITON_SCRIPT_BUILD_VERSION
fi

echo ""
echo ""
echo ""

echo -n "🐲 "
echo "Using LLVM Version based on Triton Pin: $(cat $LLVM_PIN_FILE)"
git -C $PROJECT_DIR/llvm-project reset --hard `cat $LLVM_PIN_FILE`

echo ""
echo "Setting Up VENV..."
echo ""
echo -n "🐍 "

################################################################################

pushd .
  cd $PROJECT_DIR/triton
  python3.11 -m venv .venv --prompt $VENV_PROJECT_NAME
  source .venv/bin/activate
popd

################################################################################

echo ""
echo "Pip Installing..."
echo ""
echo -n "🐍 "

# Install pip dependencies, including PyTorch Nightly, into the venv
pip3 install ninja cmake wheel scipy numpy pytest pytest-xdist pytest-forked lit pandas matplotlib pybind11 expecttest hypothesis pre-commit
pip3 install --no-cache-dir --pre torch torchvision torchaudio --index-url $TORCH_URL
# pip3 install --no-cache-dir --pre torch torchvision torchaudio fbgemm-gpu --index-url $TORCH_URL

################################################################################

echo ""
echo "Building LLVM..."
echo ""
echo -n "🐲 "

#Build LLVM
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=$LLVM_BUILD_TYPE \
  -DLLVM_CCACHE_BUILD=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_TARGETS_TO_BUILD=$LLVM_TARGETS \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DLLVM_ENABLE_PROJECTS=$LLVM_PROJECTS \
  -DCMAKE_INSTALL_PREFIX=$PROJECT_DIR/llvm-project/destdir \
  -B$LLVM_BUILD_DIR $PROJECT_DIR/llvm-project/llvm
ninja -C $LLVM_BUILD_DIR

################################################################################

echo ""
echo "Uninstalling Existing Triton Pip Installs"
echo ""
echo -n "🐍 "

# Remove Stale Tritons from poluting the Virtual Environment
pip3 uninstall -y pytorch-triton triton
pip3 uninstall -y pytorch-triton-rocm

echo ""
echo "Building Triton..."
echo ""
echo -n "⚛ "

# Build Triton
pushd .
  cd $PROJECT_DIR/triton
  DEBUG=$TRITON_BUILD_DEBUG TRITON_BUILD_WITH_CLANG_LLD=1 \
  TRITON_BUILD_WITH_CCACHE=0 \
  LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib LLVM_SYSPATH=$LLVM_BUILD_DIR \
    pip3 install -e . --no-build-isolation

  # Code Completion Setup
  ln -s ./python/build/cmake.*-cpython-* build
  ln -s ./build/compile_commands.json
popd

################################################################################

echo ""
echo ""
echo ""
echo "Now run: "
echo "export LLVM_BUILD_DIR=\`pwd\`/llvm-project/build"
echo "cd triton; source .venv/bin/activate"
echo ""
echo ""
echo ""
echo "After running the above, to rebuild Triton, run: "
echo "LLVM_INCLUDE_DIRS=\$LLVM_BUILD_DIR/include LLVM_LIBRARY_DIR=\$LLVM_BUILD_DIR/lib LLVM_SYSPATH=\$LLVM_BUILD_DIR pip3 install -e . --no-build-isolation"
echo ""
echo ""
echo ""

################################################################################

