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
  version = "2026-02-06";

  src = fetchFromGitHub {
    owner = "StanfordLegion";
    repo = "realm";
    rev = "0405b67ca14b586f7dec0dcddee194cecee7efa6";
    sha256 = "sha256-iUPVV1rh3QuyDKgXuu8aDlaZGlNwcpPvPsSVLWp8tr4=";
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
