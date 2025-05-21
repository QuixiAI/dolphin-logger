# This file is now the main entry point that delegates to the CLI.
# All substantive code has been moved to other modules within this package.

from .cli import main_cli

if __name__ == "__main__":
    main_cli()
