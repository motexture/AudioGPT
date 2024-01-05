import torch

from tqdm.auto import tqdm
from transformers import Pipeline
from transformers import BertTokenizer
from audiogpt.models import AudioGPT

class AudioGPTPipeline(Pipeline):
    def __init__(
        self,
        tokenizer: BertTokenizer,
        model: AudioGPT,
        device: str = "cuda",
        codebooks_size: int = 1024,
        n_codebooks: int = 8
    ):
        super().__init__()

        self.tokenizer = tokenizer

        self.model = model

        self.device = device

        self.model.to(device)
        self.model.eval()

        self.max_text_tokens = codebooks_size // 32
        self.codebooks_size = codebooks_size
        self.n_codebooks = n_codebooks

    def preprocess(self, text: str) -> torch.Tensor:
        text_tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_text_tokens).input_ids.unsqueeze(0).to(self.device)
        sample_length = text_tokens.size(1) * 32
        return text_tokens, sample_length

    def generate(self, text_tokens: torch.Tensor, sample_length: int, temperature: float) -> torch.Tensor:
        audio_tokens = torch.randint(1, self.codebooks_size, (1, self.n_codebooks, 1), dtype=torch.long).to(self.device)

        with torch.no_grad():
            for _ in tqdm(range(sample_length), desc="Generating tokens"):
                predicted_audio_tokens = self.model(audio_tokens, text_tokens)
                predicted_audio_tokens = predicted_audio_tokens[:, :, -1, :]

                scaled_logits = predicted_audio_tokens / temperature
                scaled_logits = scaled_logits.view(-1, scaled_logits.size(-1))

                max_logits = torch.max(scaled_logits, dim=-1, keepdim=True).values
                stable_logits = scaled_logits - max_logits

                exp_logits = torch.exp(stable_logits)
                probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)

                new_tokens = torch.multinomial(probs, num_samples=1)
                new_tokens = new_tokens.view(1, self.n_codebooks, -1)

                audio_tokens = torch.cat((audio_tokens, new_tokens), dim=-1)

        return audio_tokens

    def __call__(self, inputs: str, temperature: float = 1.0, *args, **kwargs) -> str:
        text_tokens, sample_length = self.preprocess(inputs)

        audio_tokens = self.generate(text_tokens, sample_length, temperature)

        return audio_tokens