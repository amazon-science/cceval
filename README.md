# CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion

This repository contains the data and inference code of the NeurIPS 2023  (Datasets and Benchmarks track)
paper "[CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion](https://arxiv.org/abs/2310.11248)."

## Requirements

- Uncompress the CrossCodeEval data via `tar -xvJf data/crosscodeeval_data.tar.xz -C data/`
    - The data contains {baseline, retrieval, retrieval w/ ref.} setting x {bm25, UniXCoder, OpenAI Ada} retriever.
    - **Please email us if you need the raw data.**
- Install dependencies via `pip install -r requirements.txt`
- Build tree sitter via `bash scripts/build_treesitter.sh`


## Evaluation on CrossCodeEval
Our evaluation consists of two steps: generation and metrics calculation.


### Generation

#### Open-sourced Models
For open-sourced models like StarCoder, DeepSeek-Coder, etc., we recommended using [vLLM](https://github.com/vllm-project/vllm) for fast and distributed inference on CrossCodeEval. 

```bash
export gpus=2
export model=bigcode/starcoder2-3b
export language=python
export task=line_completion_rg1_unixcoder_cosine_sim
export output_dir=./tmp/crosscodeeval_testrun/
python scripts/vllm_inference.py \
  --tp $gpus \
  --task $task \
  --language $language \
  --model $model \
  --output_dir $output_dir \
  --use_crossfile_context 
```
For additional args, e.g., cross-file context length and sampling top_p, please see `python vllm_inference.py --help`.

<details><summary> If you prefer non-vLLM script <i>:: click to expand ::</i></summary>
<div>

First, configure `accelerate` via `accelerate config` if you haven't. A reference configuration is available at `cceval_config.yaml`

The following command demonstrates how to run greedy eval using codegen-350M on python with cross-file context.

```bash
export model_type=codelm_cfc # or codelm for no cross-file context eval
export model_name=Salesforce/codegen-350M-mono
export language=python
export ts_lib=./build/${language}-lang-parser.so
export dtype=bf16 # or fp16
export prompt_file=./data/crosscodeeval_data/${language}/line_completion_rg1_unixcoder_cosine_sim.jsonl # or other options in the dir, which corresponds to different retrieval methods and/or retrieval settings
export max_seq_length=2048
export cfc_seq_length=512 
export batch_size=16 # reduce for larger models
export output_dir=./tmp/crosscodeeval_testrun/

accelerate launch eval.py \
        --model_type $model_type \
        --model_name_or_path $model_name \
        --cfc_seq_length $cfc_seq_length \
        --prompt_file $prompt_file \
        --gen_length 50 \
        --max_seq_length $max_seq_length \
        --batch_size $batch_size \
        --output_dir $output_dir \
        --dtype $dtype \
        --num_return_sequences 1 \
        --overwrite_cache True \
        --ts_lib $ts_lib \
        --language $language
```

You may run sampling via the following (additional) args:

```bash
        --do_sample \
        --top_p 0.95 \
        --temperature 0.2 \
        --num_return_sequences 5 \
```


</div>
</details>

#### OpenAI models
OpenAI models are accessible through an API. You may use the following script:
```bash
export model=gpt-3.5-turbo-0125 
export language=python
export task=line_completion_rg1_unixcoder_cosine_sim
export output_dir=./tmp/crosscodeeval_openai_testrun/
python scripts/openai_inference.py \
  --task $task \
  --language $language \
  --model $model \
  --output_dir $output_dir \
  --use_crossfile_context 

```


### Metrics Calculation
After obtaining the generation, we can calculate the final metrics
```bash
export language=python
export ts_lib=./build/${language}-lang-parser.so; 
export task=line_completion_oracle_unixcoder_cosine_sim
export prompt_file=./data/${language}/${task}.jsonl 
export output_dir=./tmp/crosscodeeval_testrun/;  
python scripts/eval.py \
  --prompt_file $prompt_file \
  --output_dir $output_dir \
  --ts_lib $ts_lib \
  --language $language \
  --only_compute_metric
```







## Citation

```
@inproceedings{ding2023crosscodeeval,
      title={CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion}, 
      author={Yangruibo Ding and Zijian Wang and Wasi Uddin Ahmad and Hantian Ding and Ming Tan and Nihal Jain and Murali Krishna Ramanathan and Ramesh Nallapati and Parminder Bhatia and Dan Roth and Bing Xiang},
      year={2023},
      booktitle={Advances in Neural Information Processing Systems},
      url={https://arxiv.org/pdf/2310.11248.pdf}
}
```
## Questions
Please feel free to email us (email addresses in the [paper](https://arxiv.org/pdf/2310.11248.pdf)). You may also submit an issue in this repo.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
