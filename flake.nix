{
  description = "bloodBender reproducible environment and run targets";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f {
            pkgs = import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
          });
    in
    {
      packages = forAllSystems ({ pkgs }:
        let
          python = pkgs.python310;
          pythonEnv = python.withPackages (ps: with ps; [
            arrow
            cryptography
            numpy
            pandas
            pyarrow
            pyjwt
            python-dotenv
            pyyaml
            pytorch-lightning
            requests
            scikit-learn
            scipy
            torch
            torchmetrics
          ]);
        in
        {
          default = pythonEnv;
          bloodbender-env = pythonEnv;

          sync-data = pkgs.writeShellApplication {
            name = "sync-data";
            runtimeInputs = [ pythonEnv ];
            text = ''
              if [[ ! -d "$PWD/bloodBath" ]]; then
                echo "ERR: Run from the bloodBender repository root (missing ./bloodBath)." >&2
                exit 1
              fi
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
              exec python -m bloodBath.cli.main production-sync "$@"
            '';
          };

          run-inference = pkgs.writeShellApplication {
            name = "run-inference";
            runtimeInputs = [ pythonEnv pkgs.bash pkgs.coreutils pkgs.findutils pkgs.gnugrep pkgs.gnused ];
            text = ''
              if [[ ! -f "$PWD/run_lstm_inference.sh" ]]; then
                echo "ERR: Run from the bloodBender repository root (missing ./run_lstm_inference.sh)." >&2
                exit 1
              fi
              export NO_VENV=1
              export BLOODBENDER_ROOT="$PWD"
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
              exec bash "$PWD/run_lstm_inference.sh" "$@"
            '';
          };
        });

      apps = forAllSystems ({ pkgs }:
        {
          sync-data = {
            type = "app";
            program = "${self.packages.${pkgs.system}.sync-data}/bin/sync-data";
          };

          run-inference = {
            type = "app";
            program = "${self.packages.${pkgs.system}.run-inference}/bin/run-inference";
          };

          default = self.apps.${pkgs.system}.sync-data;
        });

      devShells = forAllSystems ({ pkgs }:
        {
          default = pkgs.mkShell {
            packages = [
              self.packages.${pkgs.system}.bloodbender-env
            ];

            shellHook = ''
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
              echo "bloodBender nix shell active"
            '';
          };
        });
    };
}