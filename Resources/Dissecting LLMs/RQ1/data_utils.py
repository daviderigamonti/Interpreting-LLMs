import logging
import os
import re

import pandas as pd

from .constants import *

# Logging
logger = logging.getLogger(LOGGER_NAME)

def convert_bats_to_question_words(bats_path, new_file_name, skip_codes, capitalize_codes, max_synonyms=3):

    bats_elements = {}

    # Read all elements froom files and format them
    for bats_root, _, bats_files in os.walk(bats_path):
        for bats_file in bats_files: 
            if not any(skip_code in bats_file for skip_code in skip_codes):
                with open(os.path.join(bats_root, bats_file), "r", encoding="utf-8") as f:
                    category_entries = []
                    for entry in f.readlines():
                        entries = entry.strip().split("\t")
                        # If a word has acceptable synonyms, only keep the first max_synonyms
                        entries[1] = "/".join(entries[1].split("/")[:max_synonyms])
                        # If category is in capitalize_codes, capitalize its words
                        if any(cap_code in bats_file for cap_code in capitalize_codes):
                            entries = entries[0].title(), entries[1].title()
                        category_entries.append(entries)
                    bats_elements[bats_file.split(" ")[0]] = category_entries

    # Generate all possible analogy combinations between categories
    bats_elements = {
        category: [
            entry_1 + entry_2 
            for entry_1 in entries 
            for entry_2 in entries 
            if entry_1 != entry_2
        ] for category, entries in bats_elements.items()
    }

    # Print everything on a new file with the question-words notation
    with open(new_file_name, "w+", encoding="utf-8") as f:
        for category, entries in bats_elements.items():
            f.write(": " + category + "\n")
            for entry in entries:
                f.write(" ".join(entry) + "\n")


def import_question_words(
    datasets_paths, dataset_column_format, select_categories, filter_single_tokens,
    model_info=[], tokenizers=[]
):
    # Load and preprocess dataset
    lists_categories = []
    merged_dataset = []
    # Iterate over all specified datasets and merge them
    for dataset_path in datasets_paths:
        with open(dataset_path, "r", encoding="utf-8") as f:
            input_file = f.readlines()
            # Parse file format assuming no repreating categories
            for entry in input_file:
                entry = entry.strip()
                # Check if the line denotes a new category by starting with ':'
                if entry.startswith(":"):
                    category = entry[2:]
                    lists_categories.append(category)
                # Otherwise add the entry to the dataset under the last category
                else:
                    category = lists_categories[-1]
                    merged_dataset.append(entry.split(" ") + [category])

    df = pd.DataFrame(merged_dataset, columns=dataset_column_format + ["category"])

    logger.debug("Size of dataframe: %d", len(df.index))
    for category in lists_categories:
        logger.debug("%s Size: %d", category, len(df[df["category"] == category]))

    # Include only selected categories, if any, otherwise include all categories
    if len(select_categories) != 0:
        df = df[df["category"].isin(select_categories)]

    # TODO: transform all multiple alternative synonyms into their first word
    df = df.map(lambda w: tuple(w.split("/")[0])[0] if len(w.split("/")) > 1 else w)

    # Drop duplicates that may arise from merging multiple files
    df = df.drop_duplicates(["e1", "e2", "e3", "e4"])

    # Remove words that may not be embedded into a single token by all the models
    if filter_single_tokens:
        
        single_indexes = pd.Series(True, index=df.index)
        # Remove rows with 1 or more words that cannot be encoded in a single token
        for model_id, tok in tokenizers.items():
            new_single_indexes = df[dataset_column_format].apply(
                lambda row: 
                    all(
                        len(tok.tokenize(model_info[model_id]["format"](el))) == 1 and tok.tokenize(model_info[model_id]["format"](el)) != ["[UNK]"]
                        if not isinstance(el, (tuple, list)) else 
                        any(len(tok.tokenize(model_info[model_id]["format"](ell))) == 1 for ell in el)
                        for el in row
                    ),
                axis=1,
            )
            single_indexes = single_indexes & new_single_indexes
        df = df[single_indexes == True]
        # Remove alternative words that cannot be encoded in a single token
        df = df.map(
            lambda w: 
                [el for el in w if len(tok.tokenize(model_info[model_id]["format"](el))) == 1]
                if isinstance(w, (tuple, list)) else w
        )
        # Reduce single-elemet word lists to words
        df = df.map(lambda w: w[0] if isinstance(w, (tuple, list)) and len(w) == 1 else w)
        logger.debug("Size of dataframe after removing non-single-token-only analogies: %d", len(df.index))
        for category in lists_categories:
            logger.debug("%s Size: %d", category, len(df[df["category"] == category]))

    # Change underscores into spaces
    df = df.replace({"_": " "}, regex=True)
    lists_categories = [cat.replace("_", " ") for cat in lists_categories]

    # Sort BATS categories
    pattern_BATS = re.compile(r"^[A-Z]\d{2}$")
    sorted_bats = sorted([cat for cat in lists_categories if pattern_BATS.match(cat)],  key=lambda cat: (cat[0], int(cat[1:])))
    sorted_categories = []
    bats_index = 0
    for cat in lists_categories:
        if pattern_BATS.match(cat):
            sorted_categories.append(sorted_bats[bats_index])
            bats_index += 1
        else:
            sorted_categories.append(cat)

    # Rename BATS categories
    sorted_categories = [BATS_CODES_NAMES[cat] if cat in BATS_CODES_NAMES else cat for cat in sorted_categories]
    df["category"] = df["category"].replace(BATS_CODES_NAMES)

    return df, sorted_categories
