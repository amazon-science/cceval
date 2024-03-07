# Original Copyright 2021 Microsoft under MIT License.
# From https://github.com/microsoft/dpu-utils/blob/master/python/dpu_utils/codeutils/keywords/keywordlist.py

import os
import keyword
from functools import lru_cache
from typing import FrozenSet

__all__ = ['get_language_keywords']

_LANGUAGE_TO_FILENAME = {
    'c': 'c.txt',
    'cpp': 'cpp.txt',
    'c++': 'cpp.txt',
    'csharp': 'csharp.txt',
    'c_sharp': 'csharp.txt',
    'c#': 'csharp.txt',
    'go': 'go.txt',
    'java': 'java.txt',
    'javascript': 'javascript.txt',
    'js': 'javascript.txt',
    'php': 'php.txt',
    'ruby': 'ruby.txt',
    'typescript': 'typescript.txt',
    'ts': 'typescript.txt',
}


@lru_cache()
def get_language_keywords(language: str) -> FrozenSet[str]:
    """
    Returns the keywords of a programming language.

    There are some inconsistencies across languages wrt to
    what is considered a keyword. For example, the true/false
    literals are considered keywords in many languages. However,
    we exclude them here for consistency. We also exclude special
    functions-like keywords, such as `die()` in PHP.
    """
    language = language.lower()
    if language == 'python':
        return frozenset(k for k in keyword.kwlist if k != 'True' and k != 'False')
    elif language in _LANGUAGE_TO_FILENAME:
        name = _LANGUAGE_TO_FILENAME[language]
        with open(os.path.join(os.path.dirname(__file__), name)) as f:
            return frozenset(l.strip() for l in f if len(l.strip()) > 0)
    else:
        raise Exception('Language keywords `%s` not supported yet. Consider contributing it to dpu-utils.' % language)
