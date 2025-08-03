with import <nixpkgs> { };
with pkgs.python311Packages;

buildPythonPackage rec {
  pname = "asciichartpy";
  version = "1.5.25";
  format = "setuptools";

  src = fetchFromGitHub {
    owner = "kroitor";
    repo = "asciichart";
    rev = "v1.5.12";
    hash = "sha256-0FzQ0kHRrwd5Rn6FHJZx1mzXsRNnAvwNloNAo1OgDC0=";
  };

  propagatedBuildInputs = [
    setuptools
  ];

  meta = with lib; {
    changelog = "https://github.com/kroitor/asciichart/tag/${version}";
    description = "Ascii charting tool";
    homepage = "https://github.com/kroitor/asciichart";
    license = licenses.mit;
  };
}
