import contextlib
import io
import itertools
import json
import multiprocessing
import os.path as osp
import re
import signal
from collections import defaultdict
from os import environ
from typing import List, Sequence, Union

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

class TimeOutException(Exception):
    pass

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):

    def signal_handler(signum, frame):
        raise TimeOutException('Time out!')

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'

def execution(programs, timeout):
    """Execution function for running generation code.

    Args:
        programs(str): Python code to be executed.
        task_id(int): Task id of the current example.
        timeout(int): Time limit for execution, avoid unnecessary
            blocking.

    In pass@k scenario, a lot of programs should be executed.
    Some internal error cannot be handled properly, such as
    `RecursionError` might cause system break. It is better to
    separate the execution in thread or multiprocess to better
    control the process.
    """

    def _execution(programs, timeout):
        try:
            # Add exec globals to prevent the exec to raise
            # unnecessary NameError for correct answer
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(programs, exec_globals)
            key.append('pass')
        except TimeOutException:
            key.append('timeout')
        except AssertionError:
            key.append('wrong_answer')
        except BaseException as e:
            print(e)
            key.append('failed')

    manager = multiprocessing.Manager()
    key = manager.list()
    # `signal` cannot be used in child thread, therefore, we
    # need to create a process in the thread.
    p = multiprocessing.Process(target=_execution,
                                args=(programs, timeout - 1))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        # key might not have value if killed
        return 'timeout'
    return key[0]

def _process_answer(text):
    patterns = [
        r'```python\s*(.*)\s*```',
        r'```\s*(.*)\s*```',
        r'```python\s*(.*)\s*$',
        r'```\s*(.*)\s*$',
        r'(.*)\s*```.*',
    ]
    for p in patterns:
        match = re.search(p, text, re.DOTALL)
        if match:
            text = match.group(1)
            break
    text = text.split('```')[0]
    text = text.replace('\\_', '_')
    text = text.strip()
    # remove test in generation
    test_list = ['# Test', '#Test', '#test', '# test']
    for s in test_list:
        if s in text:
            text = text[:text.find(s)]
    return text

def _process_test(test_case, pred):
    formatted = pred + '\n'
    formatted += test_case
    return formatted

def compute_score(solution_str, ground_truth, extra_info):
    format_score = 0
    prediction_extract = extract_solution(solution_str)
    if prediction_extract is not None:
        format_score = 0.1
    else:
        return 0

    pred = _process_answer(prediction_extract)
    mbpp_testcase = extra_info.get('mbpp_testcase', '')
    programs = _process_test(mbpp_testcase, pred)
    key = execution(programs, 10)

    if key == 'pass':
        acc_score = 1
    else:
        acc_score = 0
    return format_score + acc_score

if __name__ == "__main__":

    data = {"text": "# Instruction\nWrite a Python program that defines a function to analyze the frequency of characters in a given string. The function should accept two parameters: the string to analyze and a boolean indicating whether the analysis should be case-sensitive. The function should return a dictionary where the keys are characters from the string and the values are the counts of those characters. If the boolean is False, the function should treat 'A' and 'a' as the same character.\n\n# Response\n```python\ndef analyze_character_frequency(text, case_sensitive):\n    frequency_dict = {}\n    if not case_sensitive:\n        text = text.lower()\n    for char in text:\n        if char in frequency_dict:\n            frequency_dict[char] += 1\n        else:\n            frequency_dict[char] = 1\n    return frequency_dict\n```", "question": "Write a Python program that defines a function to analyze the frequency of characters in a given string. The function should accept two parameters: the string to analyze and a boolean indicating whether the analysis should be case-sensitive. The function should return a dictionary where the keys are characters from the string and the values are the counts of those characters. If the boolean is False, the function should treat 'A' and 'a' as the same character.Your code should pass these tests:\n\nassert analyze_character_frequency(\"Hello, World!\", True) == {'H': 1, 'e': 1, 'l': 3, 'o': 2, ',': 1, ' ': 1, 'W': 1, 'r': 1, 'd': 1, '!': 1}\nassert analyze_character_frequency(\"Hello, World!\", False) == {'h': 1, 'e': 1, 'l': 3, 'o': 2, ',': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1, '!': 1}\nassert analyze_character_frequency(\"Mississippi\", True) == {'M': 1, 'i': 4, 's': 4, 'p': 2}\nassert analyze_character_frequency(\"Mississippi\", False) == {'m': 1, 'i': 4, 's': 4, 'p': 2}\nassert analyze_character_frequency(\"\", True) == {}", "solution": "```python\ndef analyze_character_frequency(text, case_sensitive):\n    frequency_dict = {}\n    if not case_sensitive:\n        text = text.lower()\n    for char in text:\n        if char in frequency_dict:\n            frequency_dict[char] += 1\n        else:\n            frequency_dict[char] = 1\n    return frequency_dict\n```", "entry_point": "analyze_character_frequency", "mbpp_testcase": "assert analyze_character_frequency(\"Hello, World!\", True) == {'H': 1, 'e': 1, 'l': 3, 'o': 2, ',': 1, ' ': 1, 'W': 1, 'r': 1, 'd': 1, '!': 1}\nassert analyze_character_frequency(\"Hello, World!\", False) == {'h': 1, 'e': 1, 'l': 3, 'o': 2, ',': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1, '!': 1}\nassert analyze_character_frequency(\"Mississippi\", True) == {'M': 1, 'i': 4, 's': 4, 'p': 2}\nassert analyze_character_frequency(\"Mississippi\", False) == {'m': 1, 'i': 4, 's': 4, 'p': 2}\nassert analyze_character_frequency(\"\", True) == {}"}
    prediction = data['solution']
    test_case = data['mbpp_testcase']
    prediction = _process_answer(prediction)
    programs = _process_test(test_case, prediction)
    key = execution(programs, 10)
    print(key)