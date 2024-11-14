# bscriv/ebook.py
# (c) 2024 Praxis Maldevide - MIT License granted

from bs4 import BeautifulSoup, NavigableString, Tag
import ebooklib
from ebooklib import epub
import enchant
import os
import re
from typing import Generator, List

# Initialize the English dictionary
d = enchant.Dict("en_US")

apostrophe = re.compile(r'’|`|\'|‘')
excl = re.compile(r'\!')
dash = re.compile(r'—|-')
star = re.compile(r'\*|•|■|●|‣')

def is_word(s: str) -> bool:
    """Check if a string is a valid English word."""
    return d.check(s)

def merge_split_caps(text: str) -> str:
    def merge_sequence(sequence: List[str]) -> List[str]:
        """Merge sequences of capitalized words if the concatenated word is valid."""
        i = 0
        while i < len(sequence) - 1:
            pair = sequence[i], sequence[i + 1]
            joined = ''.join(pair)
            if is_word(joined):
                sequence[i:i + 2] = [joined]  # Merge the pair into one word
            else:
                i += 1
        return sequence

    # Split the text into words, keeping punctuation attached
    words = re.findall(r"[\S',.?:!#-/\\\"]+", text)
    
    result = []
    cap_sequence: List[str] = []

    for word in words:
        #parts = re.findall(r"[\w',.?:!-\"]+|[^\w',.?:!-\"]+", word)
        parts = re.findall(r"[\w\d']+|[^\w\d']+", word)
        cap_sequence.extend(parts)
    
    if cap_sequence:
        cap_sequence = merge_sequence(cap_sequence)
        result.extend(cap_sequence)
        
    # Now, we join non words leftwards
    merged = []
    for r in result:
        if any([c.isalpha() or c.isnumeric() for c in r]):
            merged.append(r)
        else:
            last = merged.pop(-1) if merged else ''
            merged.append(last + r + ' ')
    
    return re.sub('\s\s+', ' ', ' '.join(merged))

def parse_ebook_html(ebook_path: str, try_chapter : bool = False, **kwargs) -> Generator[tuple, None, None]:
    """
    Parses the HTML content of an EPUB file, yielding only text content from each <p> block,
    while skipping specific elements with class 'calibre3' but considering valid text that follows.

    Parameters:
    - ebook_path (str): The path to the EPUB file.
    - try_chapter (bool): If True, the first paragraph of each chapter will be used to determine the chapter title.

    Returns:
    - text_generator (Generator[tuple, None, None]): A generator yielding text content.
    """
    book = epub.read_epub(ebook_path)
    basename = os.path.basename(ebook_path)
    noext = os.path.splitext(basename)[0]
    chapter_idx = 0
    paragraph_idx = 0
    cumsum_word_count = 0
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode('utf-8')
        results = list(html_tokenizer(content, **kwargs))
        if len(results) == 0:
            continue
        chapter_idx += 1
        for row in results:
            if len(row[1]) == 0:
                continue
            paragraph_idx += 1
            word_count = len((row[1]))
            cumsum_word_count += word_count
            row = [noext, paragraph_idx, chapter_idx] + list(row[:]) + [word_count, cumsum_word_count]
            yield tuple(row)

def html_tokenizer(html_content: str, try_chapter : bool = False, sep_tag : str = 'p', dump : bool = False, skipclass : str = 'calibre14', **kwargs) -> Generator[tuple, None, None]:
    """
    Generator function to tokenize HTML content, yielding text content from each <p> block.

    Parameters:
    - html_content (str): The HTML content to be tokenized.
    - try_chapter (bool): If True, the first paragraph of each chapter will be used to determine the chapter title.

    Yields:
    - text_generator (Generator[tuple, None, None]): A generator yielding text content. 
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    fix_quote = re.compile(r'“|”|»|«')
    fix_threedot = re.compile(r'(…)|(\s*[,.]\s*[,.]\s*[,.]\s*)')
    #fix_threedot = re.compile(r'…')
    # dots and circles 
    fix_threestar = re.compile(r'(\* \* \*)|(>>>)|(• • •)|(•••+)|(\*\*\*+)|(---+)|(————+)|(# # #)')
    fix_bars = re.compile(r'\|\s*\|')
    fix_spaces = re.compile(r'\s+')
    fix_periods = re.compile(r'(\w) ([,.?!])')

    def extract_and_yield_text(element, accumulated_texts: List[str]):
        if isinstance(element, NavigableString):
            accumulated_texts.append(str(element))
        elif isinstance(element, Tag):
            if element.name == 'a' and 'calibre3' in element.get('class', []):
                # Skip processing the <a class="calibre3"> tag itself, but not its siblings
                #print('skipping', element)
                return
            if element.name == 'span' and 'italic' in element.get('class', []):
                # Append italic text directly to the accumulated_texts list without yielding
                accumulated_texts.append(element.get_text())
            else:
                # Recursively process all children, including those following skipped elements
                for child in element.children:
                    extract_and_yield_text(child, accumulated_texts)

    chapter = None
    if dump:
        print(soup)
        return
        
    for i, p_tag in enumerate(soup.find_all(sep_tag)):
        accumulated_texts = []
        # if p's class is calibre14, skip it because it's metadata
        if skipclass in p_tag.get('class', []):
            #print('skipping', i)
            #continue
            pass
        else:
            #print('processing', i)
            if i == 0 and try_chapter:
                # Instead of processing, this contains our chapter and title
                markers = []
                for span in p_tag.find_all('span', class_='bold'):
                    markers.append(span.get_text())

                if len(markers) >= 2:
                    chapter = ' '.join(markers)
                continue
        
        extract_and_yield_text(p_tag, accumulated_texts)
        # if our text is '| |', skip it
        if '| |' in ' '.join(accumulated_texts):
            print('skipping | |')
            continue
        
        text = ' '.join([text.strip() for text in accumulated_texts if text.strip()])
        text = text.replace('\n', ' ')
        text = text.replace(u'\xa0', u' ')
        text = fix_quote.sub(u'"', text)
        text = apostrophe.sub(u"'", text)
        text = excl.sub(u'!', text)
        text = dash.sub(u'-', text)
        text = star.sub(u'*', text)
        text = fix_threedot.sub(u'...', text)
        text = fix_bars.sub(u'', text)
        text = fix_spaces.sub(u' ', text)
        text = fix_threestar.sub(u'---', text)
        text = fix_periods.sub(r'\1\2', text)
        text = re.sub(r'\s*-\s*', r'-', text)
        text = re.sub(r'(\w+)"(\w|\w\w)', r"\1'\2", text)
        text = merge_split_caps(text)
        text = re.sub(r'"\s*(.*?)\s*"', r'"\1"', text)
        text = re.sub(r"'\s*(.*?)\s*'", r'"\1"', text)
        text = re.sub(r'\(\s*(.*?)\s*\)', r'(\1)', text)
        text = re.sub(r'\[\s*(.*?)\s*\]', r'[\1]', text)
        text = text.strip()
        text = re.sub(r'\s*-\s*', r'-', text)
        text = re.sub(r'(\w+)"(\w|\w\w)', r"\1'\2", text)
        text = re.sub(r'(\w+)(\(.*?\))', r'\1 \2', text)
        text = re.sub(r'([\d]+)\s*([:/.\\])\s*([\d]+)\s*([:/.\\])\s*([\d]+)', r'\1\2\3\4\5', text)
        text = re.sub(r'([\w\d]+)([/]) ([\w\d]+)', r'\1\2\3', text)
        text = re.sub(r'([\d]+)([:.,\-+/=]) ([\d]+)', r'\1\2\3', text)
        text = re.sub(r'([\w]+)\s*([#])\s*([\d]+)', r'\1 \2\3', text)
        text = re.sub(r'^"\s+([^"]*?)$', r'"\1', text)
        text = re.sub(r'^\(\s+([^\(]*?)$', r'"\1', text)
        text = re.sub(r'^([B-H]|[J-Z])\s(\w+)', r'\1\2', text)
        # ://
        # TODO failure states:
        # comma delimited numbers
        # dates
        # times
        # a.m./p.m.

        if text.find('you"ll') != -1:
            print(text)
            raise Exception('you"ll')
        
        targets = [
            'Oceano', 'Generated by', 'msh-tools.com', 'bookdesigner', 'Return to Table of Contents',
            'BookDesigner', '? HYPERLINK', "? PAGE"
        ]
        skip = False
        for t in targets:
            if text.find(t) != -1:
                skip = True

        if skip:
            continue
        
        # If the first character is a capital letter, then a space, followed by more capital letters, it is likely the beginning of a chapter and needs to have the space removed
        if len(text) == 0:
            continue
        if len(text) < 4 and text.isnumeric():
            continue
        yield chapter, text

#%%
from glob import glob
import os
import re
import json
from typing import Optional

def list_incoming_books(lpath : Optional[str] = None):
    lpath = lpath or './incoming'
    for filename in glob((f'{lpath}/*.epub')):
        base = os.path.basename(filename)
        # We need to do a little more processing:
        # The potential format is
        # (<series>) Author - Title.epub
        matches = re.match(r'^\((.*?)\)\s*(.*?)\s*-\s*(.*?).epub$', base)
        if matches is not None:
            (series, author, title) = matches.groups()
            if author.find(',') != -1:
                ln, fn = author.split(',')
                author = f"{fn.strip()} {ln.strip()}"
            yield {
                'title': title.strip(),
                'series': series.strip(),
                'author': author.strip(),
                'basename': base,
                'filename': filename
            }
        else:
            metadata = os.path.splitext(base)[0].split('-')
            yield {
                'title': metadata[1].strip(),
                'author': metadata[0].strip(),
                'basename': base,
                'filename': filename
            }
    return []

#print(list(list_incoming_books("../five/incoming")))

#%%

def prepare_source():
    if not os.path.exists('./source'):
        print("Creating ./source")
        os.mkdir('./source')

    metadata = []
    for book in list_incoming_books():
        filename = book['filename']
        del book['filename']
        try:
            # slugify
            slug = re.sub(r'[^a-zA-Z0-9 ]', '', book['title'])
            slug = re.sub(r'\s+', '_', slug)
            slug = slug.lower()
            slug = slug[:64]
            metadata.append(
                {**book, 'tags': '', 'slug': slug}
            )
            # Copy the file to ./source
            os.system(f'cp "{filename}" ./source/{slug}.epub')
            #print(f"copied {filename} to {noext}.epub")
            print(slug, book['basename'])
        except OSError as e:
            print("Error: %s : %s" % (book['filename'], e.strerror))
    
    with open('./source/00_metadata.json', 'w') as outfile:
        json.dump(metadata, outfile)
    

def print_special_defaults():
    for filename in sorted(glob('./source/*.epub')):
        try:
            basename = os.path.basename(filename)
            noext = os.path.splitext(basename)[0]
            print(f"'{noext}': {{ 'try_chapter': False, 'drop': [], 'clip': [] }},")
        except OSError as e:
            print("Error: %s : %s" % (filename, e.strerror))
            

# %%
