{ lib
, stdenv
, fetchFromGitLab
, cmake
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
  pname = "legion";
  version = "2025-01-06";

  src = fetchFromGitLab {
    owner = "StanfordLegion";
    repo = "legion";
    rev = "7be1abd0207eb1126c7629b16d1123fa6f58ce9d";
    sha256 = "sha256-gTjnGYYTQwTsrV1WcY0qqpTrlwbzAPcndurRy6XnG8A=";
  };

  nativeBuildInputs = [
    cmake
  ];

  cmakeFlags = [
    "-DLegion_USE_CUDA=1"
    "-DLegion_CUDA_ARCH=${lib.concatStringsSep "," cudaCapabilities}"
    "-DLegion_MAX_DIM=${toString maxDim}"
  ];

  buildInputs = [ 
    cudatoolkit
  ];

  meta = with lib; {
    description = "Legion is a parallel programming model for distributed, heterogeneous machines";
    homepage = "https://legion.stanford.edu/";
    license = licenses.asl20;
  };
}
