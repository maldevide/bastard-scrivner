# bscriv/bw/__main__.py
# (c) 2024 Praxis Maldevide - MIT License granted

import os
import pandas as pd

from . import BookWriter, BookWriterConfig

def main():
    # Generate config
    print("Generating config...")
    config: BookWriterConfig = BookWriter.get_config()

    # Create BookWriter instance
    book_writer = BookWriter.factory(config)

    print("Loading books...")
    # Anyone with sense would parallelize this, but I'm scared of the GIL
    book_df = book_writer.load_book()

    print("First pass: Calculating index slices for each window...")
    # This returns each window-sized chunk, with the system prompt prepended.
    # And will always start with a user message
    windows = book_writer.create_strided_windows(book_df)

    print("Saving the result...")
    book_writer.save_parquet(windows)

    print("Pipeline processing complete.")


if __name__ == "__main__":
    main()
