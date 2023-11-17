# CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion

This repository contains the data and inference code of the NeurIPS 2023  (Datasets and Benchmarks track) paper "[CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion](https://arxiv.org/abs/2310.11248)."

## Requirements
- Uncompress the CrossCodeEval data via `tar -xvJf data/crosscodeeval_data.tar.xz  -C data/`
  - The data contains {baseline, retrieval, retrieval w/ ref.} setting x {bm25, UniXCoder, OpenAI Ada} retriever.
  - Please email us if you need the raw data. 
- Install dependencies via `pip install -r requirements.txt`
- Build tree sitter via `bash build_treesitter.sh`
- Configure `accelerate` via `accelerate config` if you haven't. A reference configuration is available at `cceval_config.yaml`

## Sample Command

The following command demonstrates how to run greedy eval using codegen-350M on python with cross-file context.

```bash
export model_type=codelm_cfc # or codelm for no cross-file context eval
export model_name=Salesforce/codegen-350M-mono
export lang=python
export ts_lib=./build/$lang-lang-parser.so
export dtype=bf16 # or fp16
export prompt_file=./data/crosscodeeval_data/$lang/line_completion_rg1_bm25.jsonl # or other options in the dir, which corresponds to different retrieval methods and/or retrieval settings
export max_seq_length=2048
export cfc_seq_length=512 
export batch_size=16 # reduce for larger models
export output_dir=/tmp/crosscodeeval_testrun/

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
        --language $lang
```

You may run sampling via the following (additional) args:

```bash
        --do_sample \
        --top_p 0.95 \
        --temperature 0.2 \
        --num_return_sequences 5 \
```

Additionally, please see `openai_inference.py` for OpenAI model benchmarking.
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


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
