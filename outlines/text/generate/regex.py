import collections
import math
from json import dumps
from typing import List, Optional, Tuple, Union
import pickle
import os

import interegular
import torch
from pydantic import BaseModel

from outlines.text.generate.continuation import Continuation
from outlines.text.json_schema import build_regex_from_schema
from outlines.text.parsing import find_partial_matches, map_partial_states_to_vocab


class Regex(Continuation):
    """Represents a regex-based generation model.

    `Regex` instances are constrained generation models that only generate
    sequences that match an input regex. We assume that the sequence can be
    terminated (but not necessarily) when the finite state machine corresponding
    to the regex is in an accepting state.

    >>> import outlines.text as text
    >>> sequence = text.generate.regex(model, "(0|[1-9][0-9]+)")("Return an integer between 0 and 10")

    """

    def __init__(self, model, regex_string: str, max_tokens: Optional[int]):
        super().__init__(model, max_tokens)

        #print('#### BEGIN Regex init')
        #print('Input : ')
        #print(regex_string)
        #print(max_tokens)
        vocabulary = model.tokenizer.vocabulary
        sorted_vocabulary = [
            model.tokenizer.convert_token_to_string(k, v)
            for k, v in sorted(vocabulary.items(), key=lambda kv: kv[1])
        ]

        regex_pattern = interegular.parse_pattern(regex_string)
        self.regex_fsm = regex_pattern.to_fsm().reduce()

        def partial_match_filter(string, end_idx, state_seq):
            if end_idx is not None and end_idx < len(string) - 1:
                return False
            return True


        if os.path.exists('/content/pstate_to_vocab.pkl') and os.path.exists('/content/paths.pkl'):
            with open("/content/pstate_to_vocab.pkl", "rb") as pickle_file:
                pstate_to_vocab = pickle.load(pickle_file)
            with open("/content/paths.pkl", "rb") as pickle_file:
                paths = pickle.load(pickle_file)
        else:
            pstate_to_vocab, paths = map_partial_states_to_vocab(
                list(sorted_vocabulary),
                {"REGEX": self.regex_fsm},
                partial_match_filter,
                final_state_string=model.tokenizer.eos_token,
            )
            with open("/content/pstate_to_vocab.pkl", "wb") as pickle_file:
                pickle.dump(pstate_to_vocab, pickle_file)
            with open("/content/paths.pkl", "wb") as pickle_file:
                pickle.dump(paths, pickle_file)

        # Check whether a terminal path (from the initial state of the FSM to
        # one of its terminal states) exists, raise an exception otherwise.
        traversed_states = set()
        queue = collections.deque([self.regex_fsm.initial])
        while queue:
            symbol = queue.popleft()
            for prev_state in paths["REGEX"][symbol]:
                if prev_state not in traversed_states:
                    traversed_states.add(prev_state)
                    queue.append(prev_state)

        if traversed_states.intersection(self.regex_fsm.finals) == set():
            raise ValueError(
                "The vocabulary does not allow us to build a sequence that matches the input regex"
            )

        self.pstate_to_vocab = {k: list(v) for k, v in pstate_to_vocab.items()}
        print(f'self.pstate_to_vocab: {self.pstate_to_vocab}')
        # These tuples are comprised of the FSM name, last FSM state, and
        # number of processed tokens.
        # When an EOS is observed, the last FSM state becomes `-1`.
        self.pstates: List[Tuple[str, int, int]] = []

    def create_proposal(
        self, generated_token_ids: torch.LongTensor, logits: torch.DoubleTensor, tokenizer: any = None
    ) -> torch.DoubleTensor:
        """Modify the next-token logits so that only integers can be generated.

        Parameters
        ----------
        generated_token_ids
            The token ids generated so far.
        logits
            The next-token logits.

        """
        print('#### BEGIN create_proposal')
        #print('Input : ')
        #print(generated_token_ids)
        #print(logits)
        #print('shapes')
        #print(generated_token_ids.shape)
        #print(logits.shape)
        if len(self.pstates) == 0:
            self.pstates = [
                ("REGEX", self.regex_fsm.initial, 0)
                for _ in range(generated_token_ids.shape[0])
            ]

        if generated_token_ids.shape[-1] > 0:
            new_pstates = []
            for token_seq, (_, last_fsm_state, last_token_idx) in zip(
                generated_token_ids,
                self.pstates,
            ):
                #print('token_seq')
                #print(token_seq)

                # Get the tokens we haven't already processed
                readable_tokens = token_seq[last_token_idx:]
                #print('readable_tokens_raw')
                #print(readable_tokens)
                # excluding any EOS tokens
                not_eos_mask = [
                    tk != self.model.tokenizer.eos_token_id for tk in readable_tokens
                ]
                readable_tokens = readable_tokens[not_eos_mask]
                if len(readable_tokens) > 0:
                    # If we previously ended with an EOS, we shouldn't be
                    # getting/sampling any more non-EOS tokens
                    assert last_fsm_state > -1

                    sequence = self.model.tokenizer.decode(readable_tokens)
                    print(f'readable_tokens (without current token): {readable_tokens} - {sequence}')
                    sequence = self.model.tokenizer.convert_token_to_string(sequence,readable_tokens.item())
                    print(f'readable_tokens corrected (without current token): {readable_tokens} - {sequence}')

                    ((_, state_seq),) = find_partial_matches(
                        self.regex_fsm,
                        "".join(sequence),
                        start_state=last_fsm_state,
                        verbose=True
                    )
                    pstate = (
                        "REGEX",
                        state_seq[-1],
                        last_token_idx + len(sequence),
                    )
                else:
                    pstate = ("REGEX", -1, last_token_idx)

                #print('pstate')
                #print(pstate)

                new_pstates.append(pstate)

            self.pstates = new_pstates

        masks = []
        for pstate in self.pstates:
            print(f'for {pstate} in self.pstates:')
            mask = torch.full(
                (len(self.model.tokenizer.vocabulary),), -math.inf, device=self.device
            )

            if pstate[1] > -1:

                next_support = self.pstate_to_vocab[pstate[:2]]
                print(f'pstate[0]: {pstate[0]}')
                print(f'pstate[1]: {pstate[1]}')
                print(f'pstate[:2]: {pstate[:2]}')
                print(f'next_support = self.pstate_to_vocab[pstate[:2]]: {next_support}')
            else:
                next_support = [self.model.tokenizer.eos_token_id]

            mask[next_support] = 0
            masks.append(mask.unsqueeze(0))

        mask = torch.concatenate(masks, dim=0)

        #print('Output : ')
        #print(logits)
        print('##### WITHOUT MASK')
        top_values, top_indices = torch.topk(logits, 10, dim=-1)
        print(top_values)
        print(top_indices)
        if tokenizer != None:
            print(tokenizer.decode(top_indices))

        print('##### WITH MASK')
        top_values, top_indices = torch.topk(logits + mask, 10, dim=-1)
        print(top_values)
        print(top_indices)
        if tokenizer != None:
            print(tokenizer.decode(top_indices))

        #print('shapes')
        #print(logits.shape)
        #print(mask.shape)
        #print(torch.nonzero(mask != -float('inf'), as_tuple=True))
        print('#### End create_proposal')
        return logits + mask


def regex(model, regex_string: str, max_tokens: Optional[int] = None):
    """Generate text sequences that match the input regex.

    Parameters
    ----------
    model
        The model to use to computes the next-token logits.
    regex
        The regular expression generated expressions must match.
    max_tokens
        The maximum number of tokens to generate.

    """
    return Regex(model, regex_string, max_tokens)


def integer(model, max_tokens: Optional[int] = None):
    """Generate integers.

    The regex used to constrain the generation optionally matches plus or minus
    signs and forbids leading zeros (even if the `int` function in Python allows
    them).

    Parameters
    ----------
    model
        The model to use to computes the next-token logits.
    regex
        The regular expression generated expressions must match.
    max_tokens
        The maximum number of tokens to generate.

    """
    return Regex(model, r"[-+]?\d+", max_tokens)


def float(model, max_tokens: Optional[int] = None):
    """Generate floating-point numbers.

    The regex used to constrain the generation optionally matches plus or minus
    signs, and forbids leading zeros (even if the `float` function in Python
    allows them).

    Parameters
    ----------
    model
        The model to use to computes the next-token logits.
    regex
        The regular expression generated expressions must match.
    max_tokens
        The maximum number of tokens to generate.

    """
    return Regex(model, r"([+-]?((0|[1-9]+)([.][0-9]*)?)|([.][0-9]+))", max_tokens)


def choice(model, choices: List[str], max_tokens: Optional[int] = None):
    """Choose between different sequences."""
    regex_str = r"(" + r"|".join(choices) + r")"
    return Regex(model, regex_str, max_tokens)


def json(model, schema: Union[str, BaseModel], max_tokens: Optional[int] = None):
    """Generate a text sequence that follows a JSON schema.

    Parameters
    ---------
    model
        The model to use to computes the next-token logits.
    schema
        The JSON schema, or Pydantic model, that guides the generation.
    max_tokens
        The maximum number of tokens to generate at each step.

    """
    if isinstance(schema, type(BaseModel)):
        schema = dumps(schema.model_json_schema())

    regex_str = build_regex_from_schema(schema)

    return Regex(model, regex_str, max_tokens)
