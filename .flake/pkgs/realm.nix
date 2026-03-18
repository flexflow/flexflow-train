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
  version = "2026-02-24";

  src = fetchFromGitHub {
    owner = "StanfordLegion";
    repo = "realm";
    rev = "42f7484a80e0bdacaf47d9a758822f5327348dd0";
    sha256 = "sha256-IHiokPmTjEV5df3fr1Xubuyt2N1CFI2fA7Q2TsbxS3Y=";
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
