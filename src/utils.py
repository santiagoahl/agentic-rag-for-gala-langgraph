import os, getpass

# TODO: Docstrings
# TODO: Include exception handling with link references


def get_var(var) -> None:
    if os.getenv(var):
        print(f"{var} successfully processed")
    else:
        os.environ[var] = getpass.getpass(prompt=f"Type the value of {var}: ")


def run() -> None:
    print("Hello World")


if __name__ == "__main__":
    run()
