#!/bin/bash

set -e

git submodule update --init

mkdir -p deploy
pushd deploy

function build_cmake_library {
    dep_name="$1"
    dep_url="$2"
    dep_args=("${@:3}")
    if [[ ! -e ${dep_name} ]]; then
        git clone "${dep_url}" "${dep_name}"
    fi
    if [[ ! -e "${dep_name}"_install/lib ]]; then
        mkdir -p "${dep_name}"_build "${dep_name}"_install
        pushd "${dep_name}"_build
        cmake ../"${dep_name}" -DCMAKE_INSTALL_PREFIX="$PWD"/../"${dep_name}"_install "${dep_args[@]}"
        make install -j20
        popd
    fi
    export "${dep_name}"_ROOT="$PWD"/"${dep_name}"_install
}

if [[ ! -e uv ]]; then
    mkdir uv
    XDG_BIN_HOME="$PWD"/uv sh <(curl -LsSf https://astral.sh/uv/install.sh) --no-modify-path
fi
export PATH="$PATH:$PWD/uv"

if [[ ! -e gasnet ]]; then
    git clone https://github.com/StanfordLegion/gasnet.git
fi
if [[ ! -e gasnet/release ]]; then
    make -C gasnet CONDUIT=ibv
fi
export GASNet_ROOT="$PWD"/gasnet/release

set -x

build_cmake_library Realm https://github.com/StanfordLegion/realm.git -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=ON -DREALM_ENABLE_CUDA=ON -DREALM_ENABLE_PREALM=ON -DREALM_MAX_DIM=5

build_cmake_library zstd https://github.com/facebook/zstd.git -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library benchmark https://github.com/google/benchmark.git -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON

build_cmake_library libassert https://github.com/jeremy-rifkin/libassert.git -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=ON

build_cmake_library cpptrace https://github.com/jeremy-rifkin/cpptrace.git -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=ON -DCPPTRACE_USE_EXTERNAL_ZSTD=ON

build_cmake_library NCCL https://github.com/NVIDIA/nccl.git -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

# if [[ ! -e proj ]]; then
#     git clone -b python-install https://github.com/elliottslaughter/proj.git
#     pushd proj
#     uv venv
#     uv sync
#     popd # proj
# fi
# source proj/.venv/bin/activate
# export PATH="$PATH:$PWD/proj/bin"
# export PYTHONPATH="$PYTHONPATH:$PWD/proj"

popd # deploy

mkdir build install
pushd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$PWD/../install -DFF_USE_EXTERNAL_GBENCHMARK=ON -DFF_USE_EXTERNAL_LIBASSERT=ON -DFF_USE_EXTERNAL_NCCL=ON
# proj dtgen
make install -j20
popd # build
