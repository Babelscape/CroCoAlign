<div align="center">

# <img src="media/crocodile.png" alt="img" width="30" height="30"/> CroCoAlign: A Cross-Lingual, Context-Aware and Fully-Neural Sentence Alignment System for Long Texts.

[![Conference](https://img.shields.io/badge/EACL-2024-red
)](https://2024.eacl.org)

</div>

This is the official repository for [*CroCoAlign: A Cross-Lingual, Context-Aware and Fully-Neural Sentence Alignment System for Long Texts*](https://aclanthology.org/2024.eacl-long.135/).  

![CroCoAlign](media/architecture.png "CroCoAlign Architecture")

## Citation
This work has been published at EACL 2024 (main conference). If you use any part, please consider citing our paper as follows:
```bibtex
@inproceedings{molfese-etal-2024-crocoalign,
    title = "{C}ro{C}o{A}lign: A Cross-Lingual, Context-Aware and Fully-Neural Sentence Alignment System for Long Texts",
    author = "Molfese, Francesco  and
      Bejgu, Andrei  and
      Tedeschi, Simone  and
      Conia, Simone  and
      Navigli, Roberto",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.135",
    pages = "2209--2220"
}
```

## Features

- Sentence alignment using sentence embeddings and context encoder. 
- Support for various languages.
- Customizable alignment strategies.
- Evaluation metrics for alignment quality.

## Installation

To install CroCoAlign, follow these steps:

1. Clone the repository: `git clone https://github.com/Babelscape/CroCoAlign.git`
2. Create a new conda environment: `conda create -n crocoalign python==3.9`
3. Activate the environment: `conda activate crocoalign`
4. Install the following pip version: `python -m pip install pip==24.0`
5. Install the required dependencies with: `pip install -r requirements.txt`

## Download

You can download the official checkpoint at the following [link](https://drive.google.com/file/d/1DwOAB50loUc0lBe6gImX8TI7RqxD8XCw/view).

## Preprocessing

Under the **src/sentence_aligner/preprocessing** folder you can find two python scripts.

- **dataset_generator.py** is needed to convert the original xml Opus documents into the jsonl format required for training. Under the **data** folder you can already find the preprocessed data for convenience.
- **precompute_embeddings.py** (OPTIONAL) can be used to precompute the embeddings for the data.

## Training

1. Set the variable **core.data_dir** contained in the **conf/default.yml** file to the path containing the data for train, validation and test.

2. If you want the system to compute embeddings at runtime, set the variables **conf.nn.data.precomputed_embeddings** and **conf.nn.module.precomputed_embeddings** to **False**. If you have run the **precompute_embeddings.py** script to generate embeddings for the data, you can set both the variables to **True** in order to let the system skip the sentence embedding step during the forward pass. 

3. Set the variables **conf.transformer_name** and **conf.tokenizer_transformer_name** to the desired sentence transformers using the HuggingFace name (e.g. "sentence-transformers/LaBSE") or a local path. 

3. To train a new instance of CroCoAlign, run the command:

`PYTHONPATH="src" python src/sentence_aligner/run.py param_1 ... param_n`

Where param_1 ... param_n are the parameters of the network that can be modified at runtime.  
You can consult which parameters can be changed by accessing the **conf** directory and its subdirectories.  
Alternatively, you can also modify the parameter directly in the .yaml files instead of modifying them at runtime during training. 

## Evaluation

To evaluate CroCoAlign on the Opus book dataset, run the following command:

`PYTHONPATH="src" python src/sentence_aligner/evaluate.py ckpt_path test_data_path`

You can call the script with the `-h` command to get information about the available command options.  
The data is already splitted into train, val and test and its available under the **data** folder. 

To reproduce the paper results against Vecalign, you can run the following script:

`python results/scripts/paper_results.py results/data/opus/books/crocoalign-{version}.tsv`

Where {version} needs to be replaced with one of the available recovery strategies.  
See the content of the **results/data/opus/books/** folder to see the available options.

## Inference

To align your own parallel documents using CroCoAlign, you can run the following command:

`PYTHONPATH="scr" python scr/sentence_aligner/crocoalign.py source_document target_document`

You can call the script with the `-h` command to get information about the available command options.  
The default format of the source and target document is considered to be .txt.  
In case you would like to try another sentence encoder, you can either provide a .jsonl files containing the source and target sentences with or without precomputed sentence embeddings (by using the `-p` command) or selecting another sentence encoder at runtime.  
You can also select the desired output format of the final alignment (either .tsv or .jsonl).

# License 
CroCoAlign is licensed under the CC BY-SA-NC 4.0 license. The text of the license can be found [here](https://github.com/Babelscape/CroCoAlign/blob/main/LICENSE).

We underline that the dataset we used in our experiments has been extracted from the [Opus website](https://opus.nlpl.eu/Books/corpus/version/Books), which was introduced in the following work: 
J. Tiedemann, 2012, [Parallel Data, Tools and Interfaces in OPUS](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf). In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012).

# Acknowledgments
The code in this repository is built on top of [![](https://shields.io/badge/-nn--template-emerald?style=flat&logo=github&labelColor=gray)](https://github.com/grok-ai/nn-template).
