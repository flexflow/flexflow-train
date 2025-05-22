{ fetchFromGitHub
, stdenv
, lib
, cmake
, pkg-config
, zstd
, libdwarf-lite
, libunwind
}:

stdenv.mkDerivation rec {
  pname = "cpptrace";
  version = "0.7.5";

  src = fetchFromGitHub {
    owner = "jeremy-rifkin";
    repo = "cpptrace";
    rev = "v${version}";
    sha256 = "sha256-2rDyH9vo47tbqqZrTupAOrMySj4IGKeWX8HBTGjFf+g=";
  };

  nativeBuildInputs = [
    cmake
    pkg-config
  ];

  buildInputs = [
    zstd
    libdwarf-lite
    libunwind
  ];

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=On" 
    "-DCPPTRACE_USE_EXTERNAL_ZSTD=1"
    "-DCPPTRACE_USE_EXTERNAL_LIBDWARF=1"
    "-DCPPTRACE_STD_FORMAT=0"
    "-DCPPTRACE_STATIC_DEFINE=0"
    "-DCPPTRACE_UNWIND_WITH_LIBUNWIND=1"
    "-DCPPTRACE_FIND_LIBDWARF_WITH_PKGCONFIG=1"
  ];

  meta = with lib; {
    description = "Simple, portable, and self-contained stacktrace library for C++11 and newer";
    homepage = "https://github.com/jeremy-rifkin/cpptrace";
    license = licenses.mit;
  };
}
