#!/bin/bash

set -e

if [[ -z $CI ]]; then
    module load cuda cmake
    export CC=gcc-10
    export CXX=g++-10
fi

mkdir -p deploy
pushd deploy

function build_cmake_library {
    dep_name="$1"
    dep_url="$2"
    dep_args=("${@:3}")
    if [[ ! -e ${dep_name} ]]; then
        if [[ ${dep_url} == *.git ]]; then
            git clone "${dep_url}" "${dep_name}"
        else
            mkdir "${dep_name}"
            tar xfz <(curl -LsSf "${dep_url}") -C "${dep_name}" --strip-components=1
        fi
    fi
    if [[ ! -e "${dep_name}"_install/include ]]; then
        mkdir -p "${dep_name}"_build "${dep_name}"_install
        pushd "${dep_name}"_build
        cmake ../"${dep_name}" -DCMAKE_INSTALL_PREFIX="$PWD"/../"${dep_name}"_install "${dep_args[@]}"
        make install -j20
        popd
    fi
    export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$PWD/${dep_name}_install"
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

build_cmake_library zstd https://github.com/facebook/zstd.git -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library fmt https://github.com/fmtlib/fmt/archive/refs/tags/10.2.1.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library cpptrace https://github.com/jeremy-rifkin/cpptrace/archive/refs/tags/v1.0.4.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCPPTRACE_USE_EXTERNAL_ZSTD=ON

build_cmake_library libassert https://github.com/jeremy-rifkin/libassert/archive/refs/tags/v2.2.1.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLIBASSERT_USE_EXTERNAL_CPPTRACE=ON

build_cmake_library Realm https://github.com/StanfordLegion/realm.git -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=ON -DREALM_ENABLE_CUDA=ON -DREALM_ENABLE_PREALM=ON -DREALM_ENABLE_CPPTRACE=ON -DREALM_ENABLE_HDF5=OFF -DREALM_MAX_DIM=5

build_cmake_library benchmark https://github.com/google/benchmark/archive/refs/tags/v1.9.5.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON

build_cmake_library rapidcheck https://github.com/emil-e/rapidcheck.git -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library tl-expected https://github.com/TartanLlama/expected/archive/refs/tags/v1.3.1.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library doctest https://github.com/doctest/doctest/archive/refs/tags/v2.4.12.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library spdlog https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSPDLOG_FMT_EXTERNAL=ON

build_cmake_library nlohmann_json https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library NCCL https://github.com/NVIDIA/nccl/archive/refs/tags/v2.29.7-1.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

if [[ ! -e /tmp/$USER/proj ]]; then
    git clone -b python-install https://github.com/elliottslaughter/proj.git /tmp/$USER/proj
    pushd /tmp/$USER/proj
    uv venv
    uv sync
    popd # /tmp/$USER/proj
fi
source /tmp/$USER/proj/.venv/bin/activate
export PATH="$PATH:/tmp/$USER/proj/bin"
export PYTHONPATH="$PYTHONPATH:/tmp/$USER/proj"

popd # deploy

ff_cmake_flags=(
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_INSTALL_PREFIX=$PWD/../install
    -DCMAKE_CUDA_ARCHITECTURES=60
)

proj dtgen

mkdir build install
pushd build
cmake .. "${ff_cmake_flags[@]}"
make -j20
popd # build
