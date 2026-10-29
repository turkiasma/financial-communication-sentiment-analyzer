"""Thin native-Gradio Space entry point. Importing this file never launches it."""

import os


def configure_runtime():
    # Set before importing torch/transformers/Hub; explicit user settings win.
    defaults = {"FINSENTIMENT_ALLOW_MODEL_DOWNLOAD": "1", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
                "GRADIO_ANALYTICS_ENABLED": "False", "HF_HUB_DISABLE_TELEMETRY": "1"}
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def main():
    configure_runtime()
    from src.document_processing import get_analyzer
    from src.app import launch_app
    get_analyzer()  # Initialize once at startup, using cache or the pinned public Hub revision.
    launch_app(hosted=True)


if __name__ == "__main__":
    main()
