# The Bastard Scrivner

bscriv is a set of utilities I have used for processing book data for LLM training. It provides tools to parse ebooks, tokenize their content, and prepare datasets for language model training. It is very early-stage and raw, but is prefereable to using my gists.

## Features

- Parse EPUB files and extract text content
- Tokenize HTML content from ebooks
- Process and clean extracted text
- Generate instruction-style prompts for language models
- Handle book metadata and organize processed data

## Usage

### Parsing Ebooks

To parse an ebook and extract its content:

```python
from bscriv.ebook import parse_ebook_html

for content in parse_ebook_html('path/to/your/ebook.epub'):
    print(content)
```

A much more extensive example of how to process epub books is in the samples folder [here](./samples/epub-processing.ipynb).

### Generating Training Data

To generate training data for language models:

```python
from bscriv.bw import BookWriter, BookWriterConfig

config = BookWriter.get_config()
book_writer = BookWriter.factory(config)

book_df = book_writer.load_book()
windows = book_writer.first_pass(book_df)
result_df = book_writer.second_pass(book_df, windows)
book_writer.save_parquet(result_df)
```

or

```bash
python -m bscriv.bw
```

## Configuration

You can customize the behavior of bscriv by using your own `BookWriterConfig` as defined in `bw/__init__.py`. This includes settings for data paths, output files, and tokenization parameters.

## Dependencies

bscriv relies on several libraries, including:

- pandas
- BeautifulSoup
- ebooklib
- enchant
- transformers

Make sure to install these dependencies before using bscriv.

## License

This project is licensed under the MIT License.

## Contributing

Contributions to bscriv are welcome! Please feel free to submit a Pull Request.

## Author

bscriv is created by Praxis Maldevide.

For any questions or issues, please open an issue on the GitHub repository.