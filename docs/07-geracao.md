# 07 - Geração

Depois de treinar, o modelo pode gerar texto. A geração usa o mesmo princípio do treino: prever o próximo token. A diferença é que agora o token previsto é adicionado ao contexto.

Os arquivos principais são:

- `src/srp_gpt2/inference/generator.py`
- `src/srp_gpt2/inference/sampler.py`
- `examples/generate.py`
- `src/srp_gpt2/cli.py`

## Fluxo da geração

`src/srp_gpt2/inference/generator.py::TextGenerator.generate` faz:

```text
1. Tokeniza o prompt.
2. Cria um tensor [1, T].
3. Recorta o contexto para o block_size.
4. Executa o modelo.
5. Pega os logits da última posição.
6. Escolhe o próximo token com o sampler.
7. Adiciona o token ao contexto.
8. Repete até max_new_tokens ou EOS.
```

O detalhe principal é usar apenas a última posição:

```python
next_logits = output.logits[:, -1, :]
```

Shape:

```text
output.logits: [B, T, vocab]
next_logits:   [B, vocab]
```

Na geração comum, `B=1`.

## Janela de contexto

O modelo não aceita sequência maior que `block_size`. Por isso o gerador usa:

```python
context = generated[:, -block_size:]
```

Isso mantém apenas os tokens mais recentes quando o texto gerado fica longo.

## Temperature

Temperature controla o quão aleatória é a geração.

- Valor menor que `1.0`: geração mais conservadora.
- Valor perto de `1.0`: amostragem normal.
- Valor maior que `1.0`: geração mais diversa e arriscada.

Se a temperature for muito baixa, o modelo tende a repetir escolhas óbvias. Se for alta demais, pode ficar incoerente.

## Top-k

Top-k limita a escolha aos `k` tokens com maior score.

Exemplo:

```text
top_k=50
```

O sampler ignora todos os tokens fora dos 50 mais prováveis.

## Top-p

Top-p, também chamado nucleus sampling, escolhe o menor conjunto de tokens cuja probabilidade acumulada passa de `p`.

Exemplo:

```text
top_p=0.95
```

Isso é mais adaptativo que top-k: em situações fáceis, usa poucos tokens; em situações ambíguas, permite mais opções.

## Repetition penalty

`repetition_penalty` reduz a chance de repetir tokens já gerados.

No projeto, isso fica em `src/srp_gpt2/inference/sampler.py::Sampler._apply_repetition_penalty`.

O teste `tests/test_sampler.py::test_repetition_penalty_reduces_seen_token_logits` valida esse comportamento.

## Parada por EOS

Se o tokenizador tem `eos_token_id`, o gerador pode parar quando esse token aparecer:

```python
if int(next_token.item()) == self.tokenizer.eos_token_id:
    break
```

EOS significa end of sequence, ou fim da sequência.

## Comando de geração

Com a CLI:

```bash
srp-gpt2 generate \
  --checkpoint checkpoints/gpt2-small/last.pt \
  --tokenizer gpt2 \
  --prompt "Era uma vez" \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95
```

Com o exemplo tiny:

```bash
python examples/generate.py \
  --checkpoint checkpoints/tiny/last.pt \
  --tokenizer byte \
  --prompt "Transformers usam atenção causal" \
  --max-new-tokens 30
```
