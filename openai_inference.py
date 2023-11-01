"""
Script to query an OpenAI API to generate code.
Set environment variable OPENAI_KEY with your API key
before running this script.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import openai
import tiktoken
from tqdm import tqdm

SLEEP_SECOND = 2.8          # minimum time to sleep with API errors
MAX_SLEEP_SECOND = 120      # maximum time sleep time to wait with exp backoff
MODEL_MAX_LEN = 4097        # maximum number of tokens up to which to generate
CFC_BUDGET = 512            # maximum number of tokens for cross file context
BUFFER = 20                 # estimated tokens used by OpenAI + some more buffer
SYS_PROMPT = 'You are Codex, a code completion language model. Continue the code presented to you.'
tiktokenizer = tiktoken.get_encoding('cl100k_base')
SYS_PROMPT_LEN = len(tiktokenizer.encode(SYS_PROMPT))


def query(
    prompt: str, 
    temperature: float = 0.2, 
    max_tokens: int = 128, 
    top_p: float = 1
) -> Dict:
    """
    This function queries an OpenAI API to generate code based on the given prompt.

    Args:
    prompt: str, the prompt to generate code from
    temperature: float, the value used to module the next token probabilities
    max_tokens: int, the maximum number of tokens to generate
    top_p: float, the cumulative probability for top-p filtering

    Returns:
    OpenAI Completion object, the response from the OpenAI Codex API
    """
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )


def query_with_retry(
    prompt: str, 
    temperature: float = 0.2, 
    max_tokens: int = 128, 
    top_p: float = 1
) -> Optional[Dict]:
    """
    This function queries an OpenAI API to generate code based on the given prompt.

    Args:
    prompt: str, the prompt to generate code from
    sleep_second: int, the number of seconds to sleep when the rate limit error is raised
    temperature: float, the value used to module the next token probabilities
    max_tokens: int, the maximum number of tokens to generate
    top_p: float, the cumulative probability for top-p filtering

    Returns:
    OpenAI Completion object, the response from the OpenAI Codex API if succeeds
    else return None

    Reference:
    https://github.com/Leolty/repobench/blob/c24b7a80465957e75107eafd23c66d369fa9e755/model/codex.py
    """

    error_sleep_second = SLEEP_SECOND
    def _upd_error_sleep_time(error_sleep_second):
        # double the sleep time if it is less than MAX_SLEEP_SECOND seconds
        if error_sleep_second < MAX_SLEEP_SECOND:
            error_sleep_second *= 2
        # if the sleep time is greater than MAX_SLEEP_SECOND seconds, 
        # then sleep MAX_SLEEP_SECOND seconds
        else:
            error_sleep_second = MAX_SLEEP_SECOND
        return error_sleep_second
    
    while True:
        try:
            response = query(prompt, temperature, max_tokens, top_p)
            time.sleep(SLEEP_SECOND + np.random.rand())
            return response
        except openai.error.RateLimitError as e:
            print(f'RateLimitError: {e}')
            print(f'Retrying after {error_sleep_second} seconds')
            time.sleep(error_sleep_second)
            error_sleep_second = _upd_error_sleep_time(error_sleep_second)
        except openai.error.InvalidRequestError as e:
            print(f'InvalidRequestError: {e}')
            return None
        except openai.error.OpenAIError as e:
            print(f'OpenAIError: {e}')
            print(f'Retrying after {error_sleep_second} seconds')
            time.sleep(error_sleep_second)
            error_sleep_second = _upd_error_sleep_time(error_sleep_second)


def truncate(prompt: str, max_num_tokens: int, side: str) -> str:
    """Truncate prompt from side given the token budget"""

    # use tiktokenizer to analyze num of tokens
    tokens = tiktokenizer.encode(prompt, disallowed_special=())
    num_tokens = len(tokens)

    if num_tokens > max_num_tokens:
        if side == 'left':
            prompt_tokens = tokens[num_tokens - max_num_tokens:]
        elif side == 'right':
            prompt_tokens = tokens[:max_num_tokens]
        
        # decode and encode again as a sanity check
        prompt = tiktokenizer.decode(prompt_tokens)
        new_len = len(tiktokenizer.encode(prompt, disallowed_special=()))
        assert new_len <= max_num_tokens
    return prompt


def prepare_prompt(
    prompt: str, 
    cross_file_context: str, 
    cross_file_budget: int, 
    prompt_budget: int
) -> str:
    """Create an augmented prompt according to budget specs"""

    # left truncate original prompt
    prompt = truncate(prompt, prompt_budget, 'left')

    if cross_file_context is not None:
        # right truncate cross file context string
        cross_file_context = truncate(cross_file_context, cross_file_budget, 'right')
    else:
        cross_file_context = ''
    
    # return <CFC>\n<PROMPT>
    return cross_file_context + '\n' + prompt


def get_openai_response(
    sample: Dict, 
    temperature: float, 
    new_max_tokens: int, 
    top_p: float, 
    use_cross_file: bool
) -> Tuple[str, Dict]:
    """Get OpenAI response for a single sample. Returns the prompt used to 
    infer and the response of the API."""

    if use_cross_file:
        prompt = prepare_prompt(
            sample['prompt'], sample['crossfile_context']['text'], 
            CFC_BUDGET, 
            MODEL_MAX_LEN - new_max_tokens - CFC_BUDGET - SYS_PROMPT_LEN - BUFFER
        )
    else:
        prompt = prepare_prompt(
            sample['prompt'], None, 
            0, 
            MODEL_MAX_LEN - new_max_tokens - SYS_PROMPT_LEN - BUFFER
        )
    response = query_with_retry(prompt, temperature, new_max_tokens, top_p)
    return prompt, response


def get_openai_responses(
    data: List[Dict], 
    temperature: float, 
    new_max_tokens: int, 
    top_p: float, 
    out_path: str, 
    use_cross_file: bool
) -> List[str]:
    """Get OpenAI responses to all samples in data, store in out_path,
    and return list of task ids that were skipped due to some errors"""

    skipped = []
    with open(out_path, 'w') as f:
        for d in tqdm(data):
            try:
                prompt, response = get_openai_response(
                    d, temperature, new_max_tokens, top_p, use_cross_file
                )
            except Exception as e:
                print('Unknown error', e)
                raise
            
            if response is not None:
                d['completion'] = response['choices'][0]['message']['content']
                d['api_response'] = response
                d['prompt_used'] = prompt # records the augmented prompt
                print(json.dumps(d), file=f, flush=True)
            else:
                skipped.append(d['metadata']['task_id'])
                print(f'Skipped {d["metadata"]["task_id"]}')
            
    return skipped


def main():

    # set the OpenAI key
    openai.api_key = os.environ.get('OPENAI_KEY', None)
    if openai.api_key is None:
        raise ValueError('OPENAI_KEY environment variable not set')

    # get config for current run
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument(
        '--task', type=str, required=True,
        choices=['line_completion', 'line_completion_oracle_bm25', 
        'line_completion_rg1_bm25']
    )
    parser.add_argument(
        '--language', type=str, required=True,
        choices=['csharp', 'python', 'java', 'typescript']
    )
    parser.add_argument(
        '--data_root_dir', type=str, default='data/',
        help='path to directory where data is organized in lang/task.jsonl format'
    )
    args = parser.parse_args()

    # setup paths
    data_path = os.path.join(args.data_root_dir, args.language, args.task+'.jsonl')
    data = [json.loads(l) for l in open(data_path, 'r').readlines()]
    out_path = os.path.join(
        os.path.dirname(data_path), 
        os.path.basename(data_path).split('.')[0]+'_openai_responses.jsonl'
    )
    
    # log info
    is_cross_file = 'bm25' in args.task # change logic to infer this based on use case
    print(f'data_path: {data_path}')
    print(f'number of samples: {len(data)}')
    print(f'out_path: {out_path}')
    print(f'temperature: {args.temperature}')
    print(f'max_tokens: {args.max_tokens}')
    print(f'top_p: {args.top_p}')
    print(f'is_cross_file: {is_cross_file}')
    
    # start OpenAI inference
    skipped_tasks = get_openai_responses(
        data, args.temperature, args.max_tokens, args.top_p, out_path,
        is_cross_file,
    )

    # save list of skipped tasks
    with open(out_path.replace('.jsonl', '_skipped_tasks.json'), 'w') as f:
        f.write(json.dumps(skipped_tasks))


if __name__ == '__main__':
    main()
