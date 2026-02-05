{ lib
, stdenv
, fetchFromGitHub
, cmake
, cudaPackages ? { }
, maxDim ? 5
}:

let
  inherit (cudaPackages) cudatoolkit;
in

stdenv.mkDerivation rec {
  pname = "realm";
  version = "2025-01-06";

  # This version is compatible with Legion 7be1abd0207eb1126c7629b16d1123fa6f58ce9d
  src = fetchFromGitHub {
    owner = "StanfordLegion";
    repo = "realm";
    rev = "0ef7edc8c012d4ab6a50805c044cec8a8edeae33";
    sha256 = "sha256-57/a1lAgs+ajpRn0y0Lk1gP5nKt+N08WW0DIJP4vdho=";
  };

  nativeBuildInputs = [
    cmake
  ];

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DREALM_ENABLE_CUDA=ON"
    "-DREALM_MAX_DIM=${toString maxDim}"
  ];

  buildInputs = [
    cudatoolkit
  ];

  meta = with lib; {
    description = "Realm is a distributed, event–based tasking runtime for building high-performance applications that span clusters of CPUs, GPUs, and other accelerators";
    homepage = "https://legion.stanford.edu/realm";
    license = licenses.asl20;
  };
}
