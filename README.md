# Code for the EMNLP 2025 paper "Demystifying optimized prompts in language models"

The `word-stories` model is available on [Huggingface](https://huggingface.co/rimon15/word-stories-66m)

# Citation
```bibtex
@inproceedings{melamed-etal-2025-demystifying,
    title = "Demystifying optimized prompts in language models",
    author = "Melamed, Rimon  and
      McCabe, Lucas Hurley  and
      Huang, H Howie",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.147/",
    doi = "10.18653/v1/2025.emnlp-main.147",
    pages = "2983--2999",
    ISBN = "979-8-89176-332-6",
    abstract = "Modern language models (LMs) are not robust to out-of-distribution inputs. Machine generated ({``}optimized'') prompts can be used to modulate LM outputs and induce specific behaviors while appearing completely uninterpretable. In this work, we investigate the composition of optimized prompts, as well as the mechanisms by which LMs parse and build predictions from optimized prompts. We find that optimized prompts primarily consist of punctuation and noun tokens which are more rare in the training data. Internally, optimized prompts are clearly distinguishable from natural language counterparts based on sparse subsets of the model{'}s activations. Across various families of instruction-tuned models, optimized prompts follow a similar path in how their representations form through the network."
}
```
