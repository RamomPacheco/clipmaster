import multiprocessing

# Windows + executável: tem de correr antes de qualquer spawn (Whisper em processo filho).
multiprocessing.freeze_support()


def main() -> None:
    from app.main import main as app_main  # type: ignore[import-not-found]

    app_main()


if __name__ == "__main__":
    main()
