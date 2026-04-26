"""Interactive chat session.

Single responsibility: keep a conversation history, render it through a
:class:`ChatMLTemplate`, and use a :class:`Sampler` to autoregressively
generate the assistant's next reply, stopping at ``<|im_end|>``.
"""

from __future__ import annotations

import torch
from torch import nn

from srp_gpt2.chat.template import ChatMessage, ChatMLTemplate
from srp_gpt2.chat.tokenizer import ChatTokenizer
from srp_gpt2.inference.sampler import Sampler, SamplingConfig


class ChatSession:
    """Stateful chat loop on top of a causal language model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: ChatTokenizer,
        template: ChatMLTemplate,
        device: str | torch.device = "cpu",
        system_prompt: str | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.template = template
        self.device = torch.device(device)
        self.history: list[ChatMessage] = []
        if system_prompt:
            self.history.append(ChatMessage(role="system", content=system_prompt))
        self._stop_id = tokenizer.special_id(tokenizer.specials.im_end)

    @torch.no_grad()
    def reply(
        self,
        user_message: str,
        max_new_tokens: int = 128,
        sampling: SamplingConfig | None = None,
    ) -> str:
        self.history.append(ChatMessage(role="user", content=user_message))
        prompt_ids = self.template.render_for_generation(self.history)
        generated = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        sampler = Sampler(sampling or SamplingConfig())
        block_size = self.model.config.block_size
        new_token_ids: list[int] = []

        for _ in range(max_new_tokens):
            context = generated[:, -block_size:]
            output = self.model(context)
            next_logits = output.logits[:, -1, :]
            next_token = sampler.sample(next_logits, generated)
            tok = int(next_token.item())
            if tok == self._stop_id:
                break
            new_token_ids.append(tok)
            generated = torch.cat((generated, next_token), dim=1)

        reply_text = self.tokenizer.decode(new_token_ids)
        self.history.append(ChatMessage(role="assistant", content=reply_text))
        return reply_text
