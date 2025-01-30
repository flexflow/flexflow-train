{ lib
, stdenv
, fetchFromGitLab
, cmake
, python3
, cudaPackages ? { }
, cudaCapabilities ? [ "60" "70" "80" "86" ]
, maxDim ? 5
}:

# from https://codeberg.org/Uli/nix-things/src/commit/776519e382c81b136c1d0b10d8c7b52b4acb9192/overlays/cq/python/libclang-python.nix

let 
  cmakeFlag = x: if x then "1" else "0";

  inherit (cudaPackages) cudatoolkit;
in

stdenv.mkDerivation rec {
  pname = "legion_flexflow";
  version = "2025-01-21";

  src = fetchFromGitLab {
    owner = "StanfordLegion";
    repo = "legion";
    rev = "0c5a181e59c07e3af1091a2007378ff9355047fa";
    sha256 = "sha256-oapo7klN17gmRsmaSsrpup4YJ0dtHxiKFtwz8jyPqzU=";
    fetchSubmodules = true;
  };

  nativeBuildInputs = [
    cmake
  ];

  cmakeFlags = [
    "-DLegion_USE_Python=0"
    "-DLegion_BUILD_BINDINGS=1"
    "-DLegion_USE_CUDA=1"
    "-DLegion_CUDA_ARCH=${lib.concatStringsSep "," cudaCapabilities}"
    "-DLegion_MAX_DIM=${toString maxDim}"
  ];

  buildInputs = [ 
    python3
    cudatoolkit
  ];

  meta = with lib; {
    description = "Legion is a parallel programming model for distributed, heterogeneous machines";
    homepage = "https://github.com/StanfordLegion/legion";
    license = licenses.asl20;
  };
}
