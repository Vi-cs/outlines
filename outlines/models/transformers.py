import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from outlines.models.tokenizer import Tokenizer
from outlines.params import Params

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

__all__ = ["transformers"]


class Transformers:
    """Represents a `transformers` model."""

    def __init__(
            self,
            model: "PreTrainedModel",
            tokenizer: "PreTrainedTokenizer",
            device: Optional[str] = None,
    ):
        self.device = device if device is not None else "cpu"
        self.model = model  # .to(self.device)
        self.tokenizer = tokenizer

    def __call__(
            self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        # `transformers` model accept `input_ids` of size at most equal to 2. We
        # thus reshape the input array, call the model and reshape the output
        # logits.
        if Params.verbose:
            print('#### BEGIN Transformers call')
            print('Input :')
            print(input_ids)
            print(attention_mask)
            print(self.tokenizer.decode(input_ids))

        batch_shape = input_ids.shape[:-1]
        num_tokens = input_ids.shape[-1]
        input_ids = input_ids.reshape(math.prod(batch_shape), num_tokens)

        if Params.verbose:
            print('input_ids')
            print(input_ids)
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        if Params.verbose:
            print('output')
            print(output)
        next_token_logits = output.logits[:, -1, :]
        if Params.verbose:
            print('next_token_logits')
            print(next_token_logits)

        next_token_logits = next_token_logits.reshape(batch_shape + (-1,))
        if Params.verbose:
            print('#### END Transformers call')
        return next_token_logits


class TransformersTokenizer(Tokenizer):
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, model_name: str, **kwargs):
        from transformers import AutoTokenizer

        kwargs.setdefault("padding_side", "left")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.vocabulary = self.tokenizer.get_vocab()

    def encode(
            self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: torch.LongTensor) -> List[str]:
        text = self.tokenizer.batch_decode(token_ids)
        return text

    def convert_token_to_string(self, token: str, v: int) -> List[str]:
        #print(f'convert_token_to_string: token:{token} v:{v} isinstance(v, int):{isinstance(v, int)}')

        string = self.tokenizer.convert_tokens_to_string([token])

        '''if token[0] == '▁':
            string = ' ' + string
        if len(token)>1 and token[-1] == '▁':
            string = string + ' '  '''
        #it seems that if this char is present, then they always are on the left
        if '▁' in token:
            string_firstword = " " * token.count("▁")-1 + string
            string_nextword = " " * token.count("▁") + string

            return [string_firstword,string_nextword]

        return [string]


def transformers(model_name: str, device: Optional[str] = None, **model_kwargs):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "The `transformers` library needs to be installed in order to use `transformers` models."
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = TransformersTokenizer(model_name)

    return Transformers(model, tokenizer, device)
