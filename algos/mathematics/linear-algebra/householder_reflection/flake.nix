{
  description = "Householder reflection";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonEnv = pkgs.python3.withPackages (
          ps: with ps; [
            ipython
            jax
            jaxlib
            loguru
            matplotlib
            more-itertools
            numpy
            pip
          ]
        );
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [ pythonEnv ];
        };
      }
    );
}
