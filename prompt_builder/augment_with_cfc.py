# Copyright Amazon.com, Inc. or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import time
import glob
import argparse
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from rerank_utils import lexical_ranking, SemanticReranking
from utils import str2bool, file_distance, tokenize_nltk

CHUNK_SIZE = 10
SLIDING_WINDOW_SIZE = 10  # non-overlapping chunks if SLIDING_WINDOW_SIZE=CHUNK_SIZE
QUERY_LENGTH = 10  # last N lines from prompt will be query

repository_root = "/PATH/TO/REPOS"  # get the data from authors

input_files = {
    "python": "../data/crosscodeeval_data/python/line_completion.jsonl",
    "java": "../data/crosscodeeval_data/java/line_completion.jsonl",
    "typescript": "../data/crosscodeeval_data/typescript/line_completion.jsonl",
    "csharp": "../data/crosscodeeval_data/csharp/line_completion.jsonl"
}

file_ext = {"python": "py", "java": "java", "typescript": "ts", "csharp": "cs"}


def get_crossfile_context_from_chunks(
        args,
        prompt,
        code_chunks,
        code_chunk_ids,
        groundtruth,
        semantic_ranker
):
    assert len(code_chunks) != 0
    candidate_code_chunks = code_chunks[:args.maximum_chunk_to_rerank]
    candidate_code_chunk_ids = code_chunk_ids[:args.maximum_chunk_to_rerank]

    ranking_scores = None
    meta_data = {}

    if args.rerank:
        if args.query_type == "groundtruth":
            # oracle experiment
            prompt_lines = [pl for pl in prompt.split("\n") if pl.strip()]
            groundtruth_lines = [gt for gt in groundtruth.split("\n") if gt.strip()]
            code_lines = prompt_lines + groundtruth_lines
            query = "\n".join(code_lines[-QUERY_LENGTH:])
        elif args.query_type == "last_n_lines":
            prompt_lines = [pl for pl in prompt.split("\n") if pl.strip()]
            query = "\n".join(prompt_lines[-QUERY_LENGTH:])
        else:
            raise NotImplementedError

        meta_data["query"] = query
        start = time.time()

        if args.ranking_fn == "cosine_sim":
            gpu_id = int(mp.current_process().name.split('-')[-1]) - 1
            candidate_code_chunks, candidate_code_chunk_ids, ranking_scores = semantic_ranker.rerank(
                query,
                candidate_code_chunks,
                candidate_code_chunk_ids,
                gpu_id,
                score_threshold=None
            )
        else:
            candidate_code_chunks, candidate_code_chunk_ids, ranking_scores = lexical_ranking(
                query,
                candidate_code_chunks,
                args.ranking_fn,
                candidate_code_chunk_ids,
                score_threshold=None
            )

        meta_data["latency"] = time.time() - start
        meta_data["num_candidates"] = len(candidate_code_chunks)

    top_k = min(args.maximum_cross_file_chunk, len(candidate_code_chunk_ids))
    if top_k == 0:
        return [], meta_data

    selected_chunks = []
    selected_chunks_filename = []
    selected_chunks_scores = []

    if args.use_next_chunk_as_cfc:
        # prepare an id2idx map
        assert len(candidate_code_chunks) == len(candidate_code_chunk_ids)
        id2idx = dict()
        for j, cci in enumerate(code_chunk_ids):
            id2idx[cci] = j

        total_added = 0
        for cidx, _id in enumerate(candidate_code_chunk_ids):
            fname, c_id = _id.rsplit("|", 1)
            next_id = f"{fname}|{int(c_id) + 1}"
            if next_id not in id2idx:
                to_add = code_chunks[id2idx[_id]]
            else:
                to_add = code_chunks[id2idx[next_id]]

            if to_add not in selected_chunks:
                selected_chunks.append(to_add)
                selected_chunks_filename.append(fname)
                if args.rerank:
                    selected_chunks_scores.append(ranking_scores[cidx])
                total_added += 1
                if total_added == top_k:
                    break
    else:
        selected_chunks = candidate_code_chunks[:top_k]
        selected_chunks_filename = [_id.rsplit("|", 1)[0] for _id in candidate_code_chunk_ids[:top_k]]
        if args.rerank:
            selected_chunks_scores = ranking_scores[:top_k]

    cross_file_context = []
    for idx in range(len(selected_chunks)):
        cross_file_context.append({
            "retrieved_chunk": selected_chunks[idx],
            "filename": selected_chunks_filename[idx],
            "score": selected_chunks_scores[idx] if args.rerank else None
        })

    line_start_sym = "#" if args.language == "python" else "//"
    cfc_text = f"{line_start_sym} Here are some relevant code fragments from other files of the repo:\n\n"
    for sc, scf in zip(selected_chunks, selected_chunks_filename):
        cfc_text += f"{line_start_sym} the below code fragment can be found in:\n{line_start_sym} {scf}" + "\n"
        cfc_text += "\n".join([f"{line_start_sym} {cl}" for cl in sc.strip('\n').splitlines()]) + "\n\n"

    return cross_file_context, cfc_text, meta_data


def read_project_files(repo_name, lang):
    # root_dir needs a trailing slash (i.e. /root/dir/)
    project_context = {}
    root_dir = os.path.join(repository_root, lang, repo_name)
    if not os.path.isdir(root_dir):
        print(f"Repository not found: {root_dir}")
        return project_context

    if lang == "typescript":
        src_files = []
        src_files += glob.glob(os.path.join(root_dir, f'src/**/*.ts'), recursive=True)
        src_files += glob.glob(os.path.join(root_dir, f'src/**/*.tsx'), recursive=True)
    else:
        src_files = glob.glob(os.path.join(root_dir, f'**/*.{file_ext[lang]}'), recursive=True)

    if len(src_files) == 0:
        return project_context

    for filename in src_files:
        if os.path.exists(filename):  # weird but some files cannot be opened to read
            if os.path.isfile(filename):
                try:
                    with open(filename, "r") as file:
                        file_content = file.read()
                except:
                    with open(filename, "rb") as file:
                        file_content = file.read().decode(errors='replace')

                fileid = os.path.relpath(filename, root_dir)
                project_context[fileid] = file_content
        else:
            pass
            # print(f"File not found: {filename}")

    return project_context


def find_files_within_distance_k(current_file_path, filelist, k):
    list_of_modules = []
    module_weight = []
    for filepath in filelist:
        if filepath != current_file_path:
            dist = file_distance(filepath, current_file_path)
            if dist == -1:
                continue
            elif dist <= k:
                list_of_modules.append(filepath)
                module_weight.append(dist)

    # sorting in ascending order
    list_of_modules = [x for _, x in sorted(zip(module_weight, list_of_modules))]
    return list_of_modules


def get_cfc(example, args, semantic_ranker, repositories):
    project_context = repositories[example["metadata"]["repository"]]
    status = None
    current_filepath = example["metadata"]["file"]
    if len(project_context) == 0:
        example["crossfile_context"] = ""
        status = "project_not_found"
    else:
        current_filecontent = None
        for filepath, filecontent in project_context.items():
            if filepath == current_filepath:
                current_filecontent = filecontent
                break

        if current_filecontent is None:
            example["crossfile_context"] = {}
            print(current_filepath)
            status = "file_not_found_in_project"

        else:
            pyfiles = find_files_within_distance_k(
                example["metadata"]["file"],
                list(project_context.keys()),
                k=args.crossfile_distance
            )
            pyfiles = pyfiles[:args.maximum_cross_files]

            code_chunks = []
            code_chunk_ids = []
            for pyfile in pyfiles:
                lines = project_context[pyfile].split("\n")
                lines = [l for l in lines if l.strip()]  # removing empty lines
                c_id = 0
                for i in range(0, len(lines), SLIDING_WINDOW_SIZE):
                    c = "\n".join(lines[i:i + CHUNK_SIZE])
                    tokenized_c = tokenize_nltk(c)
                    if len(tokenized_c) > 0:
                        code_chunks.append(c)
                        code_chunk_ids.append(f"{pyfile}|{c_id}")
                        c_id += 1

            if len(code_chunks) == 0:
                example["crossfile_context"] = {}
                status = "no_crossfile_context"

            else:
                cfc, cfc_text, meta_data = get_crossfile_context_from_chunks(
                    args=args,
                    prompt=example["prompt"],
                    code_chunks=code_chunks,
                    code_chunk_ids=code_chunk_ids,
                    groundtruth=example["groundtruth"],
                    semantic_ranker=semantic_ranker
                )
                example["crossfile_context"] = {}
                example["crossfile_context"]["text"] = cfc_text
                example["crossfile_context"]["list"] = cfc

    return example, status


def attach_data(args, srcfile):
    empty_cfc = 0
    error_freq = {
        "project_not_found": 0,
        "file_not_found_in_project": 0,
        "no_crossfile_context": 0
    }
    output_examples = []

    examples = []
    repositories = dict()
    with open(srcfile) as f:
        for line in f:
            ex = json.loads(line)
            repo_name = ex["metadata"]["repository"]
            if repo_name not in repositories:
                repositories[repo_name] = read_project_files(repo_name, args.language)
            examples.append(ex)

    semantic_ranker = None
    if args.ranking_fn == "cosine_sim":
        semantic_ranker = SemanticReranking(
            args.ranker,
            max_sequence_length=256
        )

    pool = mp.Pool(args.num_processes)
    worker = partial(get_cfc, args=args, semantic_ranker=semantic_ranker, repositories=repositories)

    with tqdm(total=len(examples)) as pbar:
        for (d, stat) in pool.imap_unordered(worker, examples):
            if stat in error_freq:
                error_freq[stat] += 1
            if len(d["crossfile_context"]) == 0:
                empty_cfc += 1
                if not args.skip_if_no_cfc:
                    output_examples.append(d)
            else:
                output_examples.append(d)
            pbar.update()

    print("Total examples with empty CFC: ", empty_cfc)
    print(error_freq)
    return output_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerank",
        type=str2bool,
        default=True,
        help="rerank the functions"
    )
    parser.add_argument(
        "--ranker",
        type=str,
        default="sparse",
        choices=["sparse", "unixcoder"],
        help="ranking function"
    )
    parser.add_argument(
        "--ranking_fn",
        type=str,
        default="bm25",
        choices=["tfidf", "bm25", "jaccard_sim", "cosine_sim"],
        help="ranking function"
    )
    parser.add_argument(
        "--query_type",
        type=str,
        default="last_n_lines",
        choices=["last_n_lines", "groundtruth"],
        help="how to form query from prompt"
    )
    parser.add_argument(
        "--crossfile_distance",
        type=int,
        default=100,
        help="max distance to search for crossfile"
    )
    parser.add_argument(
        "--maximum_chunk_to_rerank",
        type=int,
        default=1000,
        help="max chunks to consider to rank via BM25"
    )
    parser.add_argument(
        "--maximum_cross_files",
        type=int,
        default=1000,
        help="max chunks to consider to rank via BM25"
    )
    parser.add_argument(
        "--maximum_cross_file_chunk",
        type=int,
        default=50,
        help="max chunks to return as cfc"
    )
    parser.add_argument(
        "--use_next_chunk_as_cfc",
        type=str2bool,
        default=True,
        help="use next code chunk as context"
    )
    parser.add_argument(
        "--skip_if_no_cfc",
        type=str2bool,
        default=True,
        help="skip adding examples if there is no crossfile context"
    )
    parser.add_argument(
        "--output_file_suffix",
        type=str,
        default=None,
        help="add a suffix string to the output file"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["java", "python", "typescript", "csharp"],
        help="language name"
    )
    args = parser.parse_args()

    args.output_file_suffix = "" if args.output_file_suffix is None else f"_{args.output_file_suffix}"
    if args.use_next_chunk_as_cfc:
        assert args.rerank
        assert args.query_type != "groundtruth"

    tgtfile_suffix = ""
    if args.rerank:
        tgtfile_suffix += f"_{args.ranking_fn}"

    args.num_processes = 60
    if args.ranking_fn == "cosine_sim":
        num_gpus = 8
        args.num_processes = num_gpus
        mp.set_start_method('spawn')

    input_file = input_files[args.language]
    output_path = os.path.dirname(input_file)
    output_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = output_filename + args.output_file_suffix + tgtfile_suffix + ".jsonl"
    output_file = os.path.join(output_path, output_filename)
    output_examples = attach_data(args, input_file)
    with open(output_file, "w") as fw:
        for ex in output_examples:
            fw.write(json.dumps(ex))
            fw.write("\n")
