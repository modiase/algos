{
  description = "Disjoint Set C Implementation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        compiler = if pkgs.stdenv.isDarwin then pkgs.clang else pkgs.gcc;

        setup = pkgs.writeShellScriptBin "setup" "meson setup build";
        run = pkgs.writeShellScriptBin "run" ''
          if [ ! -f build/build.ninja ]; then
            echo "Build directory not set up, running setup first..."
            meson setup build
          fi
          redirect=""
          [ "$1" != "-v" ] && redirect=">/dev/null 2>&1"
          eval "meson compile -C build main $redirect"
          ./build/main
        '';
        run_tests = pkgs.writeShellScriptBin "run_tests" ''
          if [ ! -f build/build.ninja ]; then
            echo "Build directory not set up, running setup first..."
            meson setup build
          fi
          redirect=""
          [ "$1" != "-v" ] && redirect=">/dev/null 2>&1"
          eval "meson compile -C build $redirect"
          flags=""
          filter=""
          [ "$1" != "-v" ] && flags="--quiet" && filter="2>&1 | grep -v '^ninja:'"
          if eval "meson test -C build $flags $filter"; then
            echo "✓ All tests passed"
          else
            echo "✗ Tests failed"
          fi
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # tools
            compiler
            meson
            ninja
            pkg-config
            gnugrep

            # commands
            setup
            run
            run_tests
          ];
        };
      });
}
