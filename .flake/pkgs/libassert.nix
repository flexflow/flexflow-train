{ fetchFromGitHub
, stdenv
, lib
, cmake
, cpptrace
, zstd
}:

stdenv.mkDerivation rec {
  pname = "libassert";
  version = "2.1.4";

  src = fetchFromGitHub {
    owner = "jeremy-rifkin";
    repo = "libassert";
    rev = "v${version}";
    sha256 = "sha256-Zkh6JjJqtOf91U01fpP4hKhhXfH7YGInodG8CZxHHXQ=";
  };

  nativeBuildInputs = [
    cmake
  ];

  propagatedBuildInputs = [
    cpptrace
    zstd
  ];

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=1"
    "-DLIBASSERT_USE_EXTERNAL_CPPTRACE=1"
  ];

  meta = with lib; {
    description = "The most over-engineered C++ assertion library";
    homepage = "https://github.com/jeremy-rifkin/libassert";
    license = licenses.mit;
  };
}
