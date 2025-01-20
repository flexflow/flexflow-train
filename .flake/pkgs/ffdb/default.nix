{ lib
, stdenv
, makeWrapper
, gdb
, python3
, proj
}:

stdenv.mkDerivation rec {
  pname = "ffdb";
  version = "0.1";

  pythonPath = with python3.pkgs; makePythonPath [
    proj
  ];

  dontBuild = true;

  nativeBuildInputs = [ makeWrapper ];

  src = ./.;

  installPhase = ''
    mkdir -p $out/share/ffdb
    cp ffdb.py $out/share/ffdb
    makeWrapper ${gdb}/bin/gdb $out/bin/gdb \
      --add-flags "-q -x $out/share/ffdb/ffdb.py" \
      --set NIX_PYTHONPATH ${pythonPath} \
      --prefix PATH : ${lib.makeBinPath [
        python3
      ]}
    cp $out/bin/gdb $out/bin/ffdb
  '';

  nativeCheckInputs = [
    gdb
    python3
    proj
  ];
}
