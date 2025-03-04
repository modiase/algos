{ pkgs ? import <nixpkgs> {} }:
with pkgs;
let 
  PROJECT_ROOT = builtins.getEnv "PWD";
  asciichartpy = import "${PROJECT_ROOT}/nix/asciichartpy.nix";
  pythonPackages = ps: with ps; [ 
    asciichartpy
    autopep8
    flake8
    graphviz
    ipython
    matplotlib
    more-itertools
    numpy
    pytest
    sortedcollections
    tabulate
  ];
  py = python313.withPackages pythonPackages;
in
  mkShell {
    buildInputs = [ py ];
    shellHook = ''
    '';
  }
