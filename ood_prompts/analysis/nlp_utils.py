from lingua import Language, LanguageDetectorBuilder
import spacy
import spacy.util
import spacy.cli

from ood_prompts.utils import get_logger
from ood_prompts.config import SETUP_LANGUAGES


logger = get_logger(__name__)

LANG_TO_SPACY_MODEL = {
  Language.ENGLISH: "en_core_web_sm",
  Language.RUSSIAN: "ru_core_news_sm",
  Language.FRENCH: "fr_core_news_sm",
  Language.GERMAN: "de_core_news_sm",
  Language.SPANISH: "es_core_news_sm",
  Language.CHINESE: "zh_core_web_sm",
  Language.PORTUGUESE: "pt_core_news_sm",
  Language.KOREAN: "ko_core_news_sm",
  Language.JAPANESE: "ja_core_news_sm",
}


def download_and_load_model(lang: Language) -> spacy.language.Language:
  model_name = LANG_TO_SPACY_MODEL[lang]

  if not spacy.util.is_package(model_name):
    logger.info(f"Downloading {model_name}...")
    spacy.cli.download(model_name)

  return spacy.load(model_name)


NLP_MODELS = {Language.ENGLISH: download_and_load_model(Language.ENGLISH)}
if SETUP_LANGUAGES:
  for lang in LANG_TO_SPACY_MODEL.keys():
    NLP_MODELS[lang] = download_and_load_model(lang)

  detector = LanguageDetectorBuilder.from_languages(*list(LANG_TO_SPACY_MODEL.keys())).build()


def categorize_token(
  token: str, is_multilingual: bool = False, lang_confidence_thresh: float = 0.90
) -> tuple[str, str]:
  token = token.strip()
  if not token:
    return "Whitespace", "N/A"

  # final_lang = Language.ENGLISH
  # confidence = 0.0
  # if is_multilingual:
  #   detected_lang = detector.detect_language_of(token)
  #   if detected_lang is not None:
  #     confidence = detector.compute_language_confidence(token, detected_lang)

  #   logger.debug(f"Token: {token}; Detected language: {detected_lang}; Confidence: {confidence}")

  #   if confidence >= lang_confidence_thresh:
  #     final_lang = detected_lang

  # doc = NLP_MODELS[final_lang](token)
  # final_lang = None
  # token = doc[0]
  token = NLP_MODELS[Language.ENGLISH](token)[0]

  if token.is_punct:
    return "Punctuation", "N/A"

  tag = token.pos_
  logger.debug(f"Token: {token}; Tag: {tag}")
  tag_str = "Other"
  if tag == "NOUN" or tag == "PROPN":
    tag_str = "Noun"
  # if tag == "NOUN":
  #   return "Noun"
  # elif tag == "PROPN":
  #   return "Proper Noun"
  elif tag == "VERB":
    tag_str = "Verb"
  elif tag == "ADJ":
    tag_str = "Adjective"
  elif tag == "ADV":
    tag_str = "Adverb"
  else:
    tag_str = "Other"

  return tag_str, None  # , final_lang.name.capitalize()
