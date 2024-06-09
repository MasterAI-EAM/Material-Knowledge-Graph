# Material-Knowledge-Graph
This is the directory for paper "Construction and Application of Materials Knowledge Graph in Multidisciplinary Materials Science via Large Language Mode", which presents a automatic NLP pipeline to extract information from 150,000 peer-reviewed papers and construct the material knowledge graph, which contains more than 18,000 entities and nearly 90,000 triples.

The instructions of code:

- data
  - TrainSet1.0.json: train dataset of named entity recognition and relation extraction (NERRE) task version 1.
  - TrainSet2.0.json: train dataset of named entity recognition and relation extraction (NERRE) task version 2.
  - TrainSet_bc.json: train dataset of "Name" and "Formula" label classification (BC) task.
  - Property.xlsx and Keywords.xlsx: expert dictionary.
- [train](https://github.com/MasterAI-EAM/Darwin/blob/main/train.py): code for training LLaMA-7B (outside in main directory)
- Entity Resolution.ipynb: code for Entity Resolution (ER) task.
- Graph completion (non-embedding).ipynb: code for link predication.
- inference_KG.py: code for LLM inference


## Data Format
TrainSet{version/task}.json is the JSON file containing a list of dictionaries, and each dictionary contains the following fields:
- `instruction`: `str`, describes the task the model should perform. For NERRE, we use "You're a chatbot that can extract entities and relations in a paragraph.". For BC, we use "Tell me if the given material/chemical term belongs to the material/chemical Name or Formula.".
- `input`: `str`, input for the task.
- `output`: `str`, the answer to the instruction.

## Getting Started

1. install the requirements in the main directory:

```bash
pip install -r requirements.txt
```

Then download the checkpoints of the open-source LLaMA-7B weights from huggingface. 

2. fine-tune the LLMs using NERRE dataset and inference the corpus using inference_KG.py (FT detail shows below)
   
We use version2.0 to fine-tune the LLMs and using part of data from 1.0 for evaluation.

3. use ER code to clean the inference result and construct the KG.

[Mat2vec](https://github.com/materialsintelligence/mat2vec) and [ChemdataExactor](https://github.com/CambridgeMolecularEngineering/chemdataextractor) should be installed first.

4. use graph completion code to predicte potential links.

## Fine-tuning
To fine-tune LLaMA-7b with NERRE/BC datasets, below is a command that works on a machine with 4 A100 80G GPUs in FSDP `full_shard` mode.
Replace `<your_random_port>` with a port of your own, `<your_path_to_hf_converted_llama_ckpt_and_tokenizer>` with the
path to your converted checkpoint and tokenizer, and `<your_output_dir>` with where you want to store your outputs.
```bash
torchrun  --nproc_per_node=8 --master_port=<your_random_port> train.py \
    --model_name_or_path <your path to LLaMA-7b> \
    --data_path <your path to dataset> \
    --bf16 True \
    --output_dir <your output dir> \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 False
    --model_max_length 1024
```

To run on more gpus, you may prefer to turn down `gradient_accumulation_steps` to keep a global batch size of 128. Global batch size has not been tested for optimality.


## **Authors**

This project is a collaborative effort by the following:

UNSW: Yanpeng Ye, Shaozhou Wang, Tong Xie, Shaozhou Wang, Imran Razzak, Wenjie Zhang

CityU HK: Jie Ren, Yuwei Wan

Tongji University: Haofen Wang

All advised by Wenjie Zhang from UNSW Engineering

## **Citation**

If you use the data or code from this repository in your work, please cite it accordingly.

## **Acknowledgements**

This project has referred to the following open-source projects:

- Meta LLaMA: **[LLaMA](https://github.com/facebookresearch/llama)**
- Stanford Alpaca: **[Alpaca](https://github.com/tatsu-lab/stanford_alpaca)**
