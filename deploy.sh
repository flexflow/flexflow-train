#!/bin/bash

set -e

git submodule update --init

mkdir -p deploy
pushd deploy

if [[ ! -e gasnet ]]; then
    git clone https://github.com/StanfordLegion/gasnet.git
fi
if [[ ! -e gasnet/release ]]; then
    make -C gasnet CONDUIT=ibv
fi
export GASNet_ROOT="$PWD"/gasnet/release

set -x

if [[ ! -e realm ]]; then
    git clone https://github.com/StanfordLegion/realm.git
fi
if [[ ! -e realm_install/lib ]]; then
    mkdir -p realm_build realm_install
    pushd realm_build
    cmake ../realm -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="$PWD"/../realm_install -DBUILD_SHARED_LIBS=ON -DREALM_ENABLE_CUDA=ON -DREALM_ENABLE_PREALM=ON -DREALM_MAX_DIM=5
    make install -j20
    popd # realm_build
fi
export Realm_ROOT="$PWD"/realm_install

if [[ ! -e zstd ]]; then
    git clone https://github.com/facebook/zstd.git
fi
if [[ ! -e zstd_install/lib ]]; then
    mkdir -p zstd_build zstd_install
    pushd zstd_build
    cmake ../zstd -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="$PWD"/../zstd_install -DBUILD_SHARED_LIBS=ON
    make install -j20
    popd # zstd_build
fi
export zstd_ROOT="$PWD"/zstd_install

if [[ ! -e benchmark ]]; then
    git clone https://github.com/google/benchmark.git
fi
if [[ ! -e benchmark_install/lib ]]; then
    mkdir -p benchmark_build benchmark_install
    pushd benchmark_build
    cmake ../benchmark -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD"/../benchmark_install -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON
    make install -j20
    popd # benchmark_build
fi
export benchmark_ROOT="$PWD"/benchmark_install

if [[ ! -e libassert ]]; then
    git clone https://github.com/jeremy-rifkin/libassert.git
fi
if [[ ! -e libassert_install/lib ]]; then
    mkdir -p libassert_build libassert_install
    pushd libassert_build
    cmake ../libassert -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="$PWD"/../libassert_install
    make install -j20
    popd # libassert_build
fi
export libassert_ROOT="$PWD"/libassert_install

if [[ ! -e cpptrace ]]; then
    git clone https://github.com/jeremy-rifkin/cpptrace.git
fi
if [[ ! -e cpptrace_install/lib ]]; then
    mkdir -p cpptrace_build cpptrace_install
    pushd cpptrace_build
    cmake ../cpptrace -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="$PWD"/../cpptrace_install -DBUILD_SHARED_LIBS=ON -DCPPTRACE_USE_EXTERNAL_ZSTD=ON
    make install -j20
    popd # cpptrace_build
fi
export cpptrace_ROOT="$PWD"/cpptrace_install

popd # deploy

mkdir build install
pushd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$PWD/../install -DFF_USE_EXTERNAL_GBENCHMARK=ON -DFF_USE_EXTERNAL_LIBASSERT=ON
make install -j20
popd # build
