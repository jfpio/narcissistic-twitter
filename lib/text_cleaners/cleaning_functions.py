import re

from gensim import utils
import spacy

RE_FLAGS = re.compile("[\U0001F1E0-\U0001F1FF]")
RE_PUNCT_FOR_KEYWORDS = re.compile(r'([!"\#\$%\&\'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~])+\s', re.UNICODE)
RE_VIDEO = re.compile(r"(\[wideo])")

POLISH_STOPWORDS = set(open("resources/polish_stopwords.txt").read().split())


def uppercase(x: str) -> str:
    return x.upper()


def lowercase(x: str) -> str:
    return x.lower()


def remove_flags(txt: str) -> str:
    txt = utils.to_unicode(txt)
    return RE_FLAGS.sub("", txt)


def remove_video_tag(txt: str) -> str:
    txt = utils.to_unicode(txt)
    return RE_VIDEO.sub("", txt)


def remove_words(data: str, words_to_remove: set[str]) -> str:
    return " ".join([x for x in data.lower().split() if x not in words_to_remove])


def remove_single_characters(txt: str, list_of_chars: list[str]) -> str:
    pattern = "[" + "".join(list_of_chars) + "]"
    return re.sub(pattern, " ", txt)


def strip_punctuation_for_keywords(txt: str) -> str:
    txt = utils.to_unicode(txt)
    return RE_PUNCT_FOR_KEYWORDS.sub("", txt)


def remove_polish_stopwords(txt: str) -> str:
    return remove_words(txt, POLISH_STOPWORDS)


def take_first_paragraph(txt: str) -> str:
    return txt[: txt.find("\n\n")]


def lemmatize_en(txt: str) -> str:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(txt)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])
