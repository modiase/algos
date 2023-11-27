{ pkgs ? import <nixpkgs> {} }:
with pkgs;
let 
  PROJECT_ROOT = builtins.getEnv "PWD";
  asciichartpy = import "${PROJECT_ROOT}/nix/asciichartpy.nix";
  pythonPackages = ps: with ps; [ numpy pytest autopep8 ipython asciichartpy ];
  py = python311.withPackages pythonPackages;
in
  mkShell {
    buildInputs = [ py ];
  }
