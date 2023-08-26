import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from outlines.models.tokenizer import Tokenizer

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
        print('#### BEGIN Transformers call')
        print('Input :')
        print(input_ids)
        print(attention_mask)
        print(self.tokenizer.decode(input_ids))

        override_prompt = False
        if override_prompt:
            promptAIRBUS = '''The account_number is 8 digits long.


                    ### Input:
                    <alto><L><P PHYSICAL_IMG_NR="2"><PS><TB><TL CON="300293335593" HP="10.0" VP="10.3"/></TB><TB><TL CON="Décompte" HP="85.0" VP="309.6"/></TB><TB><TL CON="Le 30 mars 2020" HP="340.1" VP="251.3"/></TB><TB><TL CON="Achat pour votre compte le 30 mars 2020 à XPAR :" HP="85.0" VP="411.3"/></TB><TB><TL CON="Réf.: BOC07854 BL1KS0XEB7" HP="85.0" VP="674.2"/><TL CON="N° valeur: 0010953060 / ISIN: NL0000235190" HP="243.7" VP="674.2"/><TL CON="F1" HP="539.5" VP="674.2"/></TB><TB><TL CON="Cet avis ne comporte pas de signature." HP="85.0" VP="723.5"/><TL CON="Avec nos salutations distinguées." HP="416.9" VP="723.5"/></TB><TB><TL CON="Banque Lombard Odier & Cie SA" HP="85.0" VP="764.3"/><TL CON="Rue de la Corraterie 11 • 1204 Genève | Case postale 5215 • 1211 Genève 11 | Suisse" HP="85.0" VP="775.3"/><TL CON="Téléphone +41 (0)22 709 21 11 • Fax +41 (0)22 709 29 11 • www.lombardodier.com" HP="85.0" VP="786.3"/></TB><TB><TL CON="99 actions" HP="158.8" VP="429.4"/></TB><TB><TL CON="AIRBUS" HP="221.2" VP="429.3"/></TB><TB><TL CON="Au cours de EUR 67.31 (09:00:13)" HP="90.7" VP="454.2"/></TB><TB><TL CON="Montant brut EUR" HP="360.3" VP="477.2"/><TL CON="6 663.69" HP="509.3" VP="477.2"/><TL CON="Courtages" HP="369.3" VP="486.9"/><TL CON="34.98" HP="521.8" VP="486.9"/><TL CON="Timbre fédéral" HP="352.8" VP="496.7"/><TL CON="10.00" HP="521.8" VP="496.7"/></TB><TB><TL CON="Montant net EUR" HP="363.3" VP="512.2"/><TL CON="6 708.67" HP="509.3" VP="512.2"/></TB><TB><TL CON="Type d'ordre : Limite de prix" HP="90.7" VP="535.1"/></TB><TB><TL CON="Ordre initié par le client" HP="90.7" VP="558.2"/></TB><TB><TL CON="Au change de 1.058981" HP="315.3" VP="594.7"/></TB><TB><TL CON="Valeur : 1er avril 2020" HP="318.3" VP="604.7"/></TB><TB><TL CON="XPAR NYSE Euronext Paris" HP="90.7" VP="617.1"/><TL CON="EUR" HP="90.7" VP="626.8"/><TL CON="Euro" HP="127.5" VP="626.8"/><TL CON="CHF" HP="90.7" VP="636.6"/><TL CON="Franc suisse" HP="127.5" VP="636.6"/></TB><TB><TL CON="A votre débit CHF" HP="355.8" VP="650.0"/><TL CON="7 104.36" HP="509.3" VP="650.0"/></TB><TB><TL CON="599999 00 00 001MONSIEUR JOHN DOE" HP="85.0" VP="378.8"/></TB></PS></P></L></alto>'''

            additional_data = torch.tensor([[13, 1104, 29540, 2016, 29537, 7559, 15602, 24913, 13, 29516,
                                             1312, 29540, 21168, 29540, 261, 29540, 20090, 29540, 25743, 29537,
                                             29500]])
            additional_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1]])

            input_ids = torch.concatenate([self.tokenizer.encode(promptAIRBUS)[0], additional_data], axis=-1).to(
                device='cuda:0')
            attention_mask = torch.concatenate([self.tokenizer.encode(promptAIRBUS)[1], additional_mask], axis=-1).to(
                device='cuda:0')

        batch_shape = input_ids.shape[:-1]
        num_tokens = input_ids.shape[-1]
        input_ids = input_ids.reshape(math.prod(batch_shape), num_tokens)

        print('input_ids')
        print(input_ids)
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        print('output')
        print(output)
        next_token_logits = output.logits[:, -1, :]
        print('next_token_logits')
        print(next_token_logits)

        next_token_logits = next_token_logits.reshape(batch_shape + (-1,))
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

    def convert_token_to_string(self, token: str, v: int) -> str:
        print(f'convert_token_to_string: token:{token} v:{v}')



        string = self.tokenizer.convert_tokens_to_string([token])
        print(f'string:{string}')
        decoded_values = [self.tokenizer.decode(torch.tensor([[x]])) for x in v]

        if token.replace('▁', '') == string:
            return token

        #if v == 1104 or v == 29500 or v == 29570 or v == 263 or v == 6568:
        print(f'decoded_values: {decoded_values}')
        return string


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
