# bscriv/bw/__init__.py
# (c) 2024 Praxis Maldevide - MIT License granted

import json
import os
import pandas as pd
import random
import re
from typing import Generator, TypedDict
from transformers import AutoTokenizer, PreTrainedTokenizer

INS_HEADER_LEN = 5
U_ROLE = "user"
S_ROLE = "system"
A_ROLE = "assistant"

class BookWriterConfig(TypedDict):
    data_path: str
    prompt_path: str
    temp_path: str
    datafile: str
    metadatafile: str
    outfile: str
    turn_size: int
    max_window: int
    user_replacement_rate: float
    tokenizer: str

class BookWriter:
    @staticmethod
    def get_config() -> BookWriterConfig:
        return {
            "data_path": "./local",
            "temp_path": ".",
            "prompt_path": "prompt",
            "datafile": "fiction-596.parquet",
            "metadatafile": "0a_metadata.json",
            "outfile": "fiction-16k.parquet",
            "turn_size": 384,
            "max_window": 16384,
            "user_replacement_rate": 0.25,
            "tokenizer": "unsloth/Mistral-Nemo-Instruct-2407",
        }

    @classmethod
    def factory(cls, config: BookWriterConfig):
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
        book_system = json.load(open(os.path.join(config["prompt_path"], "book_system.json")))
        book_continue = json.load(open(os.path.join(config["prompt_path"], "book_continue.json")))
        book_meta = json.load(open(os.path.join(config["data_path"], config["metadatafile"])))
        return cls(config, tokenizer, book_system, book_continue, book_meta)

    def __init__(
        self,
        config: BookWriterConfig,
        tokenizer: PreTrainedTokenizer,
        book_system: list[str],
        book_continue: list[str],
        book_meta: list[dict[str, str]],
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.book_system = book_system
        self.book_continue = book_continue
        self.book_meta = book_meta
        self.book_idx = {
            r["slug"]: (r["title"], r["author"], r["tags"]) for r in self.book_meta
        }

                
    def turn_generator(self, df: pd.DataFrame) -> Generator[tuple, None, None]:
        class StateAwareStrategy:
            def __init__(self):
                self.idx = 0
                self.book = None
                self.chapter = None
                self.assistant_content = ""
                self.assistant_toks = 0
                self.last_turn_was_user = False

            @property
            def has_content(self) -> bool:
                return self.assistant_content != ""
                
            def yield_user(self, user_content: str, user_tokens: int) -> tuple:
                self.last_turn_was_user = True
                return (
                    self.book,
                    self.chapter,
                    {"role": "user", "content": user_content},
                    user_tokens + INS_HEADER_LEN,
                    len(user_content),
                )
                    
            def yield_assistant(self) -> tuple:
                self.last_turn_was_user = False
                result = [
                    self.book,
                    self.chapter,
                    {"role": "assistant", "content": self.assistant_content.strip()},
                    self.assistant_toks + INS_HEADER_LEN,
                    len(self.assistant_content),
                ]
                self.assistant_content = ""
                self.assistant_toks = 0
                return tuple(result)
            
            def set_content(self, content: str, tokens: int):
                self.assistant_content = content
                self.assistant_toks = tokens

            def append_content(self, content: str, tokens: int):
                self.assistant_content += "\n\n" + content
                self.assistant_toks += 2 + tokens

            def new_chapter(self, row) -> bool:
                if self.book is None or self.book != row["book_name"]:
                    self.book = row["book_name"]
                    self.chapter = None
                    print(f"Processing {self.book}")

                if self.chapter is None:
                    self.chapter = row["chapter_ix"] - 1
                    return True
                elif self.chapter != row["chapter_ix"] - 1:
                    self.chapter = row["chapter_ix"] - 1
                    return True
                else:
                    return False

            def sniff_row(self, row) -> bool:
                if row is None:
                    return False
                if self.chapter != row["chapter_ix"] - 1:
                    return True
                else:
                    return False

        tg = StateAwareStrategy()

        while tg.idx < len(df):

            row = df.iloc[tg.idx]

            # lookahead
            if tg.idx + 1 < len(df):
                nextrow = df.iloc[tg.idx + 1]
            else:
                nextrow = None

            if tg.new_chapter(row):
                if tg.has_content:
                    if tg.last_turn_was_user == False:
                        yield tg.yield_user("> Finish Chapter", 10)
                    yield tg.yield_assistant()

                yield tg.yield_user("> Start Chapter", 8)
                if len(row["text"]) == 0:
                    tg.idx += 1
                    continue

            if tg.last_turn_was_user == False:
                if random.random() < self.config["user_replacement_rate"] and not tg.sniff_row(nextrow):
                    user_content = self.get_first_sentence(row["text"]).strip()
                    clipped_text = row["text"][len(user_content):].strip()
                    if len(clipped_text) > 0:
                        current_assistant_content = clipped_text
                    else:
                        current_assistant_content = row["text"]
                        user_content = f"> {random.choice(self.book_continue)}"
                else:
                    user_content = f"> {random.choice(self.book_continue)}"
                    current_assistant_content = row["text"]

                user_toks = len(self.tokenizer.encode(user_content))
                yield tg.yield_user(user_content, user_toks)
            else:
                current_assistant_content = row["text"]

            current_assistant_toks = len(self.tokenizer.encode(current_assistant_content))

            if tg.assistant_toks + current_assistant_toks > self.config["turn_size"] or tg.idx == len(df) - 1:
                if tg.has_content:
                    tg.append_content(current_assistant_content, current_assistant_toks)
                    if not tg.last_turn_was_user:
                        yield tg.yield_user(f"> {random.choice(self.book_continue)}", 8)
                    yield tg.yield_assistant()
                else:
                    tg.set_content(current_assistant_content, current_assistant_toks)
            else:
                tg.append_content(current_assistant_content, current_assistant_toks)

            tg.idx += 1

        if tg.has_content:
            if not tg.last_turn_was_user:
                yield tg.yield_user(f"> Finish", INS_HEADER_LEN + 2)
            yield tg.yield_assistant()

    def load_book(self) -> pd.DataFrame:
        # Load the raw data
        df = pd.read_parquet(os.path.join(self.config["data_path"], self.config["datafile"]))
        
        # Use the turn_generator to create turns
        turns = []
        for book_name in df['book_name'].unique():
            book_df = df[df['book_name'] == book_name]
            if book_df is None:
                print(f"No data found for book: {book_name}")
                continue
            turns.extend(self.turn_generator(book_df))
        
        # Create a DataFrame from the turns
        book_df = pd.DataFrame(
            turns,
            columns=['book', 'chapter', 'turn', 'tokens', 'content_len']
        )
        
        return book_df
    
    def create_strided_windows(self, df: pd.DataFrame) -> list[dict]:
        windows = []
        current_window = []
        current_tokens = 0
        current_book = None
        chapter_offset = None
        
        for _, row in df.iterrows():
            book = row['book']
            turn = row['turn']
            tokens = row['tokens']
            chapter = row['chapter']
            
            # Start a new window if it's a new book or the window is empty
            if book != current_book or not current_window:
                if current_window:
                    windows.append({
                        'book': current_book,
                        'messages': [r[0] for r in current_window],
                        'tokens': current_tokens
                    })

                chapter_offset = chapter - 1
                chapter = chapter - chapter_offset
                
                current_book = book
                current_window = []
                current_tokens = 0
                
                system_content, system_tokens = self.get_sys(self.book_idx[book], chapter)
                current_window = [
                    ({'role': 'system', 'content': system_content}, system_tokens + INS_HEADER_LEN)
                ]
                current_tokens = system_tokens + INS_HEADER_LEN
            else:
                # Adjust chapter number based on the offset
                if chapter_offset is None:
                    chapter_offset = chapter - 1
                    
                chapter = chapter - chapter_offset
            
            # Check if adding this turn would exceed the max window size
            if current_tokens + tokens > self.config['max_window']:
                # if the last turn is a user turn, we don't want to include it in our window
                # so we remove it and add it to the next window
                
                ut = None
                if current_window[-1][0]['role'] == 'user':
                    ut = current_window.pop()
                    current_tokens -= ut[1]
                
                # Save the current window and start a new one
                windows.append({
                    'book': current_book,
                    'messages': [r[0] for r in current_window],
                    'tokens': current_tokens
                })
                
                system_content, system_tokens = self.get_sys(self.book_idx[book], chapter)
                
                # Find the last assistant turn
                overlap_start = len(current_window) - 1
                while overlap_start > 0 and current_window[overlap_start][0]['role'] != 'assistant':
                    overlap_start -= 1
                
                # If we found an assistant turn, start from the user turn before it
                if overlap_start > 0:
                    overlap_start -= 1
                
                turns = current_window[overlap_start:]
                if ut is not None:
                    turns.append(ut)
                current_window = [
                    ({'role': 'system', 'content': system_content}, system_tokens + INS_HEADER_LEN),
                    *turns
                ]
                current_tokens = sum(t[1] for t in current_window)
            
            # Add the current turn to the window
            current_window.append((turn, tokens))
            current_tokens += tokens
        
        # Add the last window if it's not empty
        if len(current_window) > 0:
            if current_window[-1][0]['role'] == 'user':
                ut = current_window.pop()
                current_tokens -= ut[1]
            if len(current_window) > 0:
                windows.append({
                    'book': current_book,
                    'messages': [r[0] for r in current_window],
                    'tokens': current_tokens
                })
        
        return pd.DataFrame(windows, columns=['book', 'messages', 'tokens'])

    def get_sys(self, meta: tuple[str, str, str], chapter: int) -> tuple[str, int]:
        system_format = """{0}
Tags: {3}
Author: {2}
Title: {1}
Chapter: {4}
"""
        system = random.choice(self.book_system)
        system_content = system_format.format(system, meta[0], meta[1], meta[2], chapter)
        sys_toks = len(self.tokenizer.encode(system_content)) + INS_HEADER_LEN
        return system_content, sys_toks

    def get_first_sentence(self, text: str) -> str:
        sentences = re.split(r'([.?!]+"?)', text, 1)
        return ''.join(sentences[0:2]).strip() if len(sentences) > 0 else text

    def save_parquet(self, df: pd.DataFrame):
        df.to_parquet(self.config["outfile"])