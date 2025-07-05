import json
import os
import re
from os import environ
from tqdm import tqdm
import copy
from math_verify import parse, verify


def extract_solution(solution_str: str, reward_manager: str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<｜Assistant｜>" in solution_str:
        processed_str = solution_str.split("<｜Assistant｜>", 1)[1]
    else:
        if reward_manager == 'prime':
            processed_str = solution_str
        else:
            print("[Error] Failed to locate model response header")
            return None, solution_str

    if '</think>' in processed_str:
        final_answer = processed_str.split('</think>')[-1].lstrip().rstrip('<｜end▁of▁sentence｜>')
    else:
        final_answer = None
    return final_answer, processed_str
    
def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(pred_str, strip_double_curly_brace=True):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    # final_answer = final_answer.split('=')[-1]
    SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''),
                     (r'\ ', ''), (' ', ''), ('mbox', 'text'),
                     (',\\text{and}', ','), ('\\text{and}', ','),
                     ('\\text{m}', '\\text{}'), ('\\le', '<')]
    REMOVED_EXPRESSIONS = [
        'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
        'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
        'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals',
        'edges', 'students', 'childrentickets', 'multiples', '\\text{s}',
        '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
        '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
        '{,}', '"', '\\dots', '\n', '\r', '\f'
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(\\text\{)\((.*?)\)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    assert '\n' not in final_answer
    assert '\r' not in final_answer
    assert '\f' not in final_answer
    if len(re.findall(r'finalansweris(.*)', final_answer)) > 0:
        final_answer = re.findall(r'finalansweris(.*)', final_answer)[-1]

    if len(re.findall(r'answer?is:?(.*)', final_answer)) > 0:
        final_answer = re.findall(r'answer?is:?(.*)', final_answer)[-1]

    if len(re.findall(r'oxed\{(.*?)\}', final_answer)) > 0:
        final_answer = re.findall(r'oxed\{(.*?)\}', final_answer)[-1]

    if len(re.findall(r'\$(.*?)\$', final_answer)) > 0:
        final_answer = re.findall(r'\$(.*?)\$', final_answer)[-1]
    final_answer = final_answer.strip()
    if 'rac' in final_answer and '\\frac' not in final_answer:
        final_answer = final_answer.replace('rac', '\\frac')

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if len(substr) > 0 and substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except AssertionError:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing
    # units
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string

def _fix_sqrt_v2(string):
    _string = re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)
    return _string

def _strip_string(string):
    # linebreaks
    string = string.replace('\n', '')

    # remove inverse spaces
    string = string.replace('\\!', '')

    # replace \\ with \
    string = string.replace('\\\\', '\\')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
    # add "0" if "." is the start of the string
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(' ', '')

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works
    # with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix
    # in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def _strip_string_v2(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace('\n', '')

    # right "."
    string = string.rstrip('.')

    # remove inverse spaces
    string = string.replace('\\!', '')
    string = string.replace('\\ ', '')

    # replace \\ with \
    string = string.replace('\\\\', '\\')
    string = string.replace('\\\\', '\\')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r'\\text{.*?}$', '', string).strip()
    if _string != '' and _string != string:
        string = _string

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')
    string = string.replace('$', '')

    string = string.replace('\\text', '')
    string = string.replace('x\\in', '')

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')  # noqa: W605
    string = string.replace('%', '')

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
    # add "0" if "." is the start of the string
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')

    # cdot
    string = string.replace('\\cdot', '')

    # inf
    string = string.replace('infinity', '\\infty')
    if '\\infty' not in string:
        string = string.replace('inf', '\\infty')
    string = string.replace('+\\inity', '\\infty')

    # and
    string = string.replace('and', '')
    string = string.replace('\\mathbf', '')

    # use regex to remove \mbox{...}
    string = re.sub(r'\\mbox{.*?}', '', string)

    # quote
    string.replace("'", '')
    string.replace('"', '')

    # i, j
    if 'j' in string and 'i' not in string:
        string = string.replace('j', 'i')

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r'(\d+)\.0+([^\d])', r'\1\2', string)
    string = re.sub(r'(\d+)\.0+$', r'\1', string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    string = _fix_sqrt_v2(string)
    string = string.replace(' ', '')

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple
    # cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, version='v2', verbose=False):
    if str1 is None and str2 is None:
        print('WARNING: Both None')
        return 1.0
    if str1 is None or str2 is None:
        return 0.0

    if version == 'v1':
        strip_string_func = _strip_string
    elif version == 'v2':
        strip_string_func = _strip_string_v2
    else:
        raise NotImplementedError

    try:
        ss1 = strip_string_func(str1)
        ss2 = strip_string_func(str2)
        if verbose:
            print(ss1, ss2)
        if ss1 == ss2:
            return 1.0
        ss1 = normalize_final_answer(ss1)
        ss2 = normalize_final_answer(ss2)
        if ss1 == ss2:
            return 1.0
    except Exception:
        pass
    try:
        ss1 = normalize_final_answer(str1)
        ss2 = normalize_final_answer(str2)
        if ss1 == ss2:
            return 1.0
    except Exception:
        pass
    
    # return str1 == str2
    if str1 == str2:
        return 1.0
    else:
        if verify(parse(str1),parse(str2)):
            return 1.0
    return 0.0


def math_postprocess_v2(text: str) -> str:
    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        return cand_ans
    else:
        return text

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    # print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        # print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            # print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if positions['think_start'] > positions['think_end']:
        # print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        pass
        # print("  Tag sequence validation passed")

    return validation_passed

def compute_score(solution_str, ground_truth, reward_manager='naive'):
    prediction_extract, processed_str  = extract_solution(solution_str, reward_manager)
    # Validate response structure
    solution_text = ground_truth.get('target', '')
    if reward_manager == 'naive':
        format_correct = validate_response_structure(processed_str)
        format_score = 1 if format_correct else 0
    else:
        format_score = 1

    if format_score and prediction_extract:
        prediction_result = math_postprocess_v2(prediction_extract)
        if is_equiv(prediction_result, solution_text):
            return 1
        else:
            if '<｜end▁of▁sentence｜>' not in solution_str:
                return  0.1
            return 0
    else:
        return 0

if __name__ == "__main__":
    str1 = "The volume of the cube is calculated by \\(4^3 = 64\\) cubic feet. The volume of the cylindrical section removed has a radius of 2 feet and a height of 2 feet, which is computed as \\(\\pi \\times 2^2 \\times 2 = 8\\pi\\) cubic feet. Subtracting the volume of the cylinder from the cube's volume gives \\(64 - 8\\pi\\). Thus, the total remaining volume is \\(\\boxed{64 - 8\\pi}\\)."
    str2="64-8\\pi"
    str1_convert='$195$'
    str2='195'
    # str1_convert = math_postprocess_v2(str1)
    result = is_equiv(str1_convert, str2)
    print(result) #1.0

