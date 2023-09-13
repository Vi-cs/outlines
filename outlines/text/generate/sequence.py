from typing import List, Optional, Tuple, Union

import torch
from outlines.params import Params


class Sequence:
    """Represents a sequence generation method."""

    def __init__(self, model, max_tokens: Optional[int] = None):
        """Create a `Sequence` instance.

        Parameters
        ----------
        model
            The instance of the model used to generate next-token probabilities.
        max_tokens
            The maximum number of tokens that will be generated if no termination
            condition is met.

        """
        self.model = model
        self.device = model.device
        self.max_tokens = max_tokens
        self.pad_token_id = torch.tensor(
            model.tokenizer.pad_token_id, device=model.device
        )

    def create_proposal(
            self, generated_token_ids: torch.LongTensor, logits: torch.DoubleTensor, tokenizer: any = None
    ) -> torch.DoubleTensor:
        """Create a new proposal from the next-token logits."""
        return logits

    def is_finished(self, token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Determine whether we should stop the generation."""
        raise NotImplementedError(
            "`Sequence.is_finished` must be implemented by subclasses."
        )

    def postprocess_completions(self, completions: List[str]) -> List[str]:
        return completions

    def step(
            self,
            rng: torch.Generator,
            num_prompt_tokens: int,
            token_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            samples: int = 1,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """Generate one or several tokens that complete the input sequence.

        The sampling step consists in using a model to generate next-token
        logits and then sample `samples`-many new tokens from a categorical
        distribution parametrized by these logits.

        Parameters
        ----------
        rng
            NumPy random number Generator instance
        num_prompt_tokens
            The number of tokens in the prompt.
        token_ids
            The token ids passed as an input to the model, of shape `batch_shape
            + (num_tokens,)`, where `num_tokens` is the sequences' length.
        samples
            The number of continuations to sample from the next-token probability
            distribution.

        Returns
        -------
        A tuple with an array of shape `new_batch_shape + (num_tokens+1,)`that
        contains the completed sequences (input token ids and generated token
        ids) and an array of shape `new_batch_shape + (vocab_size,)` that
        contains the next token probabilities.
        `new_batch_shape` is computed by removing dimensions of size one in
        `(samples,) + batch_shape`.

        """
        if Params.verbose:
            print('######################### BEGIN step')
            # print('Input : ')
            # print(str(num_prompt_tokens))
            # print(str(token_ids))
            # print(attention_mask)

        num_input_dims = token_ids.ndim

        probs = self.model(token_ids, attention_mask)
        if Params.verbose:
            print('token_ids[:, num_prompt_tokens:]')
            print(token_ids[:, num_prompt_tokens:])
            print('decoded token_ids[:, num_prompt_tokens:]')
            print(self.model.tokenizer.decode(token_ids[:, num_prompt_tokens:]))

        if Params.verbose:
            print(f'Tokens: {token_ids[:, num_prompt_tokens:]}')
            print(f'Output: {self.model.tokenizer.decode(token_ids[:, num_prompt_tokens:])}')

        probs = self.create_proposal(token_ids[:, num_prompt_tokens:], probs, self.model.tokenizer)
        probs = torch.nn.functional.softmax(probs, dim=-1)
        # print('torch.nn.functional.softmax(probs, dim=-1)')
        # print(probs)

        # Sample `samples`-many new tokens
        next_token_ids = vectorized_random_choice(rng, probs, samples, self.model.tokenizer)
        # print('next_token_ids')
        # print(next_token_ids)

        # Add the missing `num_tokens` and `num_sample` dimensions
        next_token_ids = torch.unsqueeze(next_token_ids, -1)
        token_ids = torch.unsqueeze(token_ids, 0)

        # print('decoded next_token_ids')
        # print(self.model.tokenizer.decode(next_token_ids))

        # Expand the input `token_ids` array to be able to concatenate several
        # samples.
        if samples > 1:
            repetitions = (samples,) + (1,) * num_input_dims
            token_ids = torch.tile(token_ids, repetitions)
            probs = torch.tile(probs, repetitions)

        token_ids = torch.concatenate([token_ids, next_token_ids], axis=-1)

        # Merge sample and batch dimensions by removing dimensions of length
        # 1. The shape of the resulting arrays is `new_batch_shape + (num_tokens,)`
        # and `new_batch_shape + (vocab_size,)` respectively.
        token_ids = torch.atleast_2d(token_ids.squeeze())
        probs = torch.atleast_2d(probs.squeeze())

        # print('Output : ')
        # print(str(token_ids))
        # print(str(probs))
        # print('#### END step')
        return token_ids, probs

    def expand_attention_mask(
            self, attention_mask: torch.LongTensor
    ) -> torch.LongTensor:
        """Expand the attention mask after the last completion."""
        batch_shape = attention_mask.shape[:-1]
        attention_mask = torch.concatenate(
            [
                attention_mask,
                torch.broadcast_to(
                    torch.tensor([1], device=self.device), batch_shape + (1,)
                ),
            ],
            axis=-1,
        )
        return attention_mask

    def update_token_ids(
            self,
            is_finished: torch.BoolTensor,
            token_ids: torch.LongTensor,
            token_ids_unfinished: torch.LongTensor,
    ) -> torch.LongTensor:
        """Update the array of token ids after the last completion.

        We only generate new tokens for the sequences that are not finished. We thus
        update the array with the new tokens, and append pad tokens to the finished
        sequences.

        Parameters
        ----------
        is_finished
            Boolean array that indicates which sequences are finished.
        token_ids
            Array that contains the sequences before the generation's last step.
        token_ids_unfinished
            Array that contains the sequences of the unfinished sequences
            after the generation's last step.

        Returns
        -------
        An array that contains the updated array that contains the sequences. We append
        pad tokens to the finished sequences.

        """
        batch_shape = token_ids.shape[:-1]
        num_tokens = token_ids.shape[-1]
        new_token_ids = torch.empty(
            batch_shape + (num_tokens + 1,), dtype=torch.int64, device=self.device
        )
        token_ids_finished = torch.concatenate(
            [
                token_ids[is_finished],
                torch.broadcast_to(
                    self.pad_token_id,
                    token_ids[is_finished].shape[:-1] + (1,),
                ),
            ],
            axis=-1,
        )

        new_token_ids[~is_finished] = token_ids_unfinished
        new_token_ids[is_finished] = token_ids_finished

        return new_token_ids

    @torch.inference_mode()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            samples: int = 1,
            rng: Optional[torch.Generator] = None,
    ) -> Union[str, List[str]]:
        """Generate a new sequence given a prompt.

        Parameters
        ----------
        prompt
            The input prompt.
        samples
            The number of samples to generate for each prompt.

        Returns
        -------
        The full sequence that contains the prompts and the generated string.

        """
        token_ids, attention_mask = self.model.tokenizer.encode(prompt)

        token_ids = token_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if rng is None:
            rng = torch.Generator(device=self.device)

        num_prompt_tokens = token_ids.shape[-1]

        if samples > 1:
            token_ids, _ = self.step(
                rng, num_prompt_tokens, token_ids, attention_mask, samples
            )
            is_finished = self.is_finished(token_ids)

            num_batch_dims = token_ids.ndim - 1
            repetitions = (samples,) + (1,) * num_batch_dims
            attention_mask = torch.tile(attention_mask, repetitions)
            attention_mask = self.expand_attention_mask(attention_mask)
        else:
            batch_shape = token_ids.shape[:-1]
            is_finished = torch.zeros(batch_shape, dtype=torch.bool, device=self.device)

        while True:
            if Params.verbose:
                print('###### sequence.__call__()')
            num_generated_tokens = token_ids.shape[-1] - num_prompt_tokens
            if torch.all(is_finished) or num_generated_tokens == self.max_tokens:
                break

            updated_token_ids, _ = self.step(
                rng,
                num_prompt_tokens,
                token_ids[~is_finished],
                attention_mask[~is_finished],
            )
            token_ids = self.update_token_ids(is_finished, token_ids, updated_token_ids)
            attention_mask = self.expand_attention_mask(attention_mask)
            is_finished[~is_finished] = self.is_finished(
                updated_token_ids[:, num_prompt_tokens:]
            ).flatten()


            result_temp = self.model.tokenizer.decode(token_ids[..., num_prompt_tokens:])
            result_temp = self.postprocess_completions(result_temp)

            ##TODO better handling of spaces before :
            result_temp_quickfixed = [string.replace(' :', ':') for string in result_temp]
            print(f'Step output: {result_temp_quickfixed}')


            # print('CALL decode : ')
            # print(str(result_temp))
            # result_temp = self.postprocess_completions(result_temp)
            # print('CALL postprocess_completions : ')
            # print(str(result_temp))

        result = self.model.tokenizer.decode(token_ids[..., num_prompt_tokens:])
        result = self.postprocess_completions(result)

        if len(result) == 1:
            return result[0]

        ##TODO better handling of spaces before :
        result = [string.replace(' :', ':') for string in result]
        return result


def vectorized_random_choice(
        rng: torch.Generator,
        p: torch.FloatTensor,
        samples: int = 1,
        tokenizer: any = None
):
    """Vectorized implementation of `np.random.choice`.

    `np.random.choice` does not support arrays of probability. This implements
    the equivalent of this function where the `p` argument can be a matrix.

    Note
    ----
    `torch.searchsorted` may be more efficient, but it is not implemented for
    every backend, for instance MPS.

    Parameters
    ----------
    rng
        Torch random number Generator instance
    p
        An array of probability of shape `(num_probability_vectors, num_items)`
        that must sum to 1.
    samples
        The number of samples to take for each probability vector.

    Returns
    -------
    An array of shape `(num_samples, batch_size)`

    """
    cumsum = torch.unsqueeze(p.cumsum(axis=-1), 0)
    if Params.verbose:
        print('vectorized_random_choice - cumsum')
        print(cumsum)
    rand = torch.rand(
        (samples,) + p.shape[:-1] + (1,), generator=rng, device=rng.device
    )
    if Params.verbose:
        print('vectorized_random_choice - rand')
        print(rand)
    idx = (cumsum < rand).sum(axis=-1)
    if Params.verbose:
        print('vectorized_random_choice - idx')
        print(idx)

    # return idx
    # Use torch.argmax to find the index of the maximum probability along the last axis.
    max_indices = torch.argmax(p, dim=-1)
    max_indices = torch.unsqueeze(max_indices, dim=0)
    if Params.verbose:
        print('vectorized_random_choice - max_indices')
        print(max_indices)

        if max_indices != idx:
            print('DIFF TOKEN')

    top_values, top_indices = torch.topk(p, 10, dim=-1)
    if Params.verbose:
        print(top_values)
        print(top_indices)

        if tokenizer != None:
            print(tokenizer.decode(max_indices))
            print(tokenizer.decode(idx))
            print(tokenizer.decode(top_indices))

    return max_indices
