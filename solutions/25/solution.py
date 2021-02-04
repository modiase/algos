
from collections import deque
import logging
from typing import Deque, NamedTuple, Optional, Union

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(lineno)d : %(message)s")

class Attempt(NamedTuple):
    matched_string : str
    remaining_string : str
    remaining_pattern : str

class DecisionPoint(NamedTuple):
    greedy : Attempt
    non_greedy : Attempt

def advance_attempt(attempt: Attempt, greedy = False) -> Attempt:
        char = attempt.remaining_pattern[0]

        if greedy and char!= '*':
            raise RuntimeError("Can't perform a greedy advancement on a non wildcard char.")

        new_matched_string = attempt.matched_string + char
        if greedy:
            new_remaining_pattern = attempt.remaining_pattern
        else:
            new_remaining_pattern = attempt.remaining_pattern[1:]
        new_remaining_string = attempt.remaining_string[1:]

        return Attempt(matched_string=new_matched_string,
                        remaining_pattern=new_remaining_pattern,
                        remaining_string=new_remaining_string)

def next_char_is_match(next_char : str, string : str) -> bool:
    """Returns true if the supplied next_char in the pattern matches the 
    head of the string. Next char is expected to be a single char or an 
    empty string."""

    if next_char == '':
        return False
    if next_char == '*':
        return True
    if next_char == '.' and len(string) != 0:
        return True
    return next_char == string[0]
    

def take_next_greedy(attempt: Attempt) -> Attempt:
    return advance_attempt(attempt=attempt,greedy=True)

def take_next_non_greedy(attempt : Attempt) -> Attempt:
    return advance_attempt(attempt)

def take_next(attempt : Attempt) -> Optional[Union[DecisionPoint,Attempt]]:
    """Returns None if no next match is possible; otherwise, returns either 
    a single progression of the attempt — i.e., a single character moved into
    the matched field from the remaining field — or, in the case of a wildcard,
    both the greedy and non-greedy next possibilities."""


    if (len(attempt.remaining_string) == 0 \
        or len(attempt.remaining_pattern) == 0):
        return None
    next_char_to_match_in_pattern = attempt.remaining_pattern[0]
    next_char_to_match_in_string = attempt.remaining_string[0]
    
    if next_char_to_match_in_pattern == '*':
        return DecisionPoint(greedy=take_next_greedy(attempt),
                             non_greedy=take_next_non_greedy(attempt))

    if (next_char_to_match_in_pattern == next_char_to_match_in_string\
        or next_char_to_match_in_pattern == '.'):
        return advance_attempt(attempt)
    
    
    
def main_loop(queue : Deque[Attempt]) -> bool:
        logging.debug(queue)
        logging.info(len(queue))

        attempt = queue.popleft()
        logging.info(attempt)

        if attempt.remaining_pattern == ''\
            and attempt.remaining_string == '':
            return True

        next_step = take_next(attempt)

        if next_step is None:
            return False
        elif isinstance(next_step,DecisionPoint):
            queue.appendleft(next_step.non_greedy)
            queue.appendleft(next_step.greedy)
            return False
        else:
            queue.appendleft(next_step)
            return False

def main(pattern : str, string : str) -> bool:
    q = deque()
    q.append(Attempt(matched_string='',
                     remaining_string=string,
                     remaining_pattern=pattern))
    match_found = False
    while len(q) != 0 and not match_found:
        match_found = main_loop(q)
        
    logging.info(match_found) 
    return match_found

if __name__ == '__main__':
    main('.*at*.rq','chatsdafrzafafrq')
    
    


    


