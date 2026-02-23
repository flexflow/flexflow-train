{ lib
, stdenv
, fetchFromGitHub
, cmake
, cudaPackages ? { }
, zlib
, maxDim ? 5
}:

let
  inherit (cudaPackages) cudatoolkit;
in

stdenv.mkDerivation rec {
  pname = "realm";
  version = "2026-02-22-prealm";

  src = fetchFromGitHub {
    owner = "StanfordLegion";
    repo = "realm";
    rev = "6ab01f413926a2428c3c799a345f69b4807d5595";
    sha256 = "sha256-MN8nJ9O6oCZbbrE/ROvIlogtXJiSLsVZxoVXJUTeSHs=";
  };

  nativeBuildInputs = [
    cmake
  ];

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DREALM_ENABLE_CUDA=ON"
    "-DREALM_ENABLE_PREALM=ON"
    "-DREALM_MAX_DIM=${toString maxDim}"
  ];

  buildInputs = [
    cudatoolkit
    zlib
  ];

  meta = with lib; {
    description = "Realm is a distributed, event–based tasking runtime for building high-performance applications that span clusters of CPUs, GPUs, and other accelerators";
    homepage = "https://legion.stanford.edu/realm";
    license = licenses.asl20;
  };
}
