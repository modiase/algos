{ pkgs ? import <nixpkgs> {} }:
with pkgs;
let 
  PROJECT_ROOT = builtins.getEnv "PWD";
  asciichartpy = import "${PROJECT_ROOT}/nix/asciichartpy.nix";
  pythonPackages = ps: with ps; [ 
    more-itertools numpy pytest autopep8 ipython asciichartpy tabulate flake8  graphviz sortedcollections
  ];
  py = python311.withPackages pythonPackages;
in
  mkShell {
    buildInputs = [ py ];
    shellHook = ''
    '';
  }
