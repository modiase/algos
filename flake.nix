{
  description = "Algorithms repository development environment";

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

        pythonPackages =
          ps: with ps; [
            pytest
            click
            loguru
            numpy
            matplotlib
            seaborn
            pandas
            jax
            jaxlib
            pyvis
            graphviz
            more-itertools
            tabulate
          ];

        python = pkgs.python313.withPackages pythonPackages;

        setupVenv = pkgs.writeShellScriptBin "setup-python-venv" ''
                    set -e
                    VENV_DIR=".venv"
                    RECREATE=false

                    for arg in "$@"; do
                      case $arg in
                        --recreate) RECREATE=true ;;
                      esac
                    done

                    if [[ -d "$VENV_DIR" ]]; then
                      if [[ "$RECREATE" == "true" ]]; then
                        echo "Recreating virtualenv at $VENV_DIR..."
                        rm -rf "$VENV_DIR"
                      else
                        echo "Virtualenv already exists at $VENV_DIR (use --recreate to rebuild)"
                        exit 0
                      fi
                    fi

                    echo "Setting up Python virtualenv at $VENV_DIR..."
                    mkdir -p "$VENV_DIR/bin" "$VENV_DIR/lib/python3.13"

                    ln -sf ${python}/bin/python3 "$VENV_DIR/bin/python"
                    ln -sf ${python}/bin/python3 "$VENV_DIR/bin/python3"

                    SITE_PACKAGES=$(${python}/bin/python3 -c "import site; print(site.getsitepackages()[0])")
                    ln -sf "$SITE_PACKAGES" "$VENV_DIR/lib/python3.13/site-packages"

                    cat > "$VENV_DIR/pyvenv.cfg" << EOF
          home = ${python}/bin
          include-system-site-packages = false
          version = 3.13
          EOF

                    echo "Done! Virtualenv created at $VENV_DIR"
                    echo "Python: ${python}/bin/python3"
                    echo "Site-packages: $SITE_PACKAGES"
        '';

      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            setupVenv
            pkgs.pre-commit
          ];

          shellHook = ''
            echo "Python development environment loaded"
            echo "Run 'setup-python-venv' to create .venv for IDE integration"
          '';
        };

        apps.pythonDev = {
          type = "app";
          program = "${setupVenv}/bin/setup-python-venv";
        };

        packages.python = python;
      }
    );
}
