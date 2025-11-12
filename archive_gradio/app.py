import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import gradio as gr

# --- Environment and Initial Setup ---
# This must be the first import to set up environment variables
from src.infrastructure.config.settings import setup_environment

setup_environment()

# Import the database initializer
from src.infrastructure.persistence.database import init_db

# Import the UI creator from the presentation layer
from src.presentation.ui import create_ui


def main():
    """
    Main function to initialize the database, create the UI, and launch the application.
    """
    # Initialize the database (creates and loads data if db doesn't exist)
    # Note: init_db() also calls load_data_to_db()
    init_db()

    # Create the Gradio UI by calling the function from the UI module
    demo = create_ui()

    # Launch the application
    # The allowed_paths are necessary for Gradio to serve local images from these directories
    demo.launch(allowed_paths=["assets", "temp_img", "best_images_and_icons"])


if __name__ == "__main__":
    main()
