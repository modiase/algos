{ pkgs ? import <nixpkgs> {} }:
with pkgs;
let 
  pythonPackages = ps: with ps; [ numpy pytest autopep8 ];
  py = python311.withPackages pythonPackages;
in
  mkShell {
    buildInputs = [ py ];
  }
