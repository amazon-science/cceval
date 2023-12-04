#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

function generate_data() {
    lang=$1
    ranker=$2
    ranking_fn=$3

    echo "$lang, $ranker, $ranking_fn"

    output_file_suffix=""
    if [[ $ranker != "sparse" ]]; then
        output_file_suffix="_${ranker}"
    fi

    # for RG-1
    python augment_with_cfc.py \
        --language $lang \
        --rerank True \
        --ranker $ranker \
        --ranking_fn $ranking_fn \
        --query_type last_n_lines \
        --crossfile_distance 100 \
        --maximum_chunk_to_rerank 1000 \
        --maximum_cross_files 1000 \
        --maximum_cross_file_chunk 5 \
        --use_next_chunk_as_cfc True \
        --skip_if_no_cfc False \
        --output_file_suffix "rg1${output_file_suffix}"

    # for oracle experiment
    python augment_with_cfc.py \
        --language $lang \
        --rerank True \
        --ranker $ranker \
        --ranking_fn $ranking_fn \
        --query_type groundtruth \
        --crossfile_distance 100 \
        --maximum_chunk_to_rerank 1000 \
        --maximum_cross_files 1000 \
        --maximum_cross_file_chunk 5 \
        --use_next_chunk_as_cfc False \
        --skip_if_no_cfc False \
        --output_file_suffix "oracle${output_file_suffix}"
}

generate_data python sparse bm25
generate_data java sparse bm25
generate_data typescript sparse bm25
generate_data csharp sparse bm25

generate_data python sparse jaccard_sim
generate_data java sparse jaccard_sim
generate_data typescript sparse jaccard_sim
generate_data csharp sparse jaccard_sim

generate_data python unixcoder cosine_sim
generate_data java unixcoder cosine_sim
generate_data typescript unixcoder cosine_sim
generate_data csharp unixcoder cosine_sim
