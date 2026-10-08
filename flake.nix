{
  description = "C++ partial differential equation framework";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    dealii.url = "git+https://codeberg.org/landinjm/dealii-flake.git";
  };

  outputs = {flake-parts, ...} @ inputs:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      perSystem = {
        pkgs,
        system,
        ...
      }: {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            pkg-config

            # Main packages
            cmake
            gnumake
            ninja
            gcc
            mpi
            inputs.dealii.packages.${system}.dealii-980-dbg
          ];

          shellHook = ''
            echo "Dev shell loaded!"
          '';
        };
      };
    };
}
