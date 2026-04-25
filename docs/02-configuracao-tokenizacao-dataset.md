# 02 - Configuração, tokenização e dataset

Antes de construir o Transformer, o projeto define três coisas:

- Quais hiperparâmetros o modelo usa.
- Como texto vira IDs numéricos.
- Como o dataset cria exemplos para prever o próximo token.

Essas peças ficam em:

- `src/srp_gpt2/config.py`
- `src/srp_gpt2/data/tokenizer.py`
- `src/srp_gpt2/data/dataset.py`

## Configuração

`src/srp_gpt2/config.py::ModelConfig` guarda os hiperparâmetros da arquitetura:

```python
ModelConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    bias=True,
)
```

O significado principal:

- `vocab_size`: quantidade de tokens possíveis.
- `block_size`: tamanho máximo do contexto.
- `n_layer`: número de blocos Transformer.
- `n_head`: número de cabeças de atenção.
- `n_embd`: tamanho do vetor interno de cada token.
- `dropout`: regularização durante treino.
- `bias`: se camadas lineares e LayerNorm usam bias.

Um detalhe importante: `n_embd` precisa ser divisível por `n_head`. Se `n_embd=768` e `n_head=12`, cada head recebe `64` dimensões.

`TrainingConfig` guarda opções do treino, como batch size, learning rate, gradient accumulation e intervalos de log/checkpoint. `DataConfig` guarda opções do dataset, como `stride`.

## Tokenização

Modelos de linguagem não recebem texto cru. Eles recebem IDs inteiros.

No projeto, a interface mínima fica em `src/srp_gpt2/data/tokenizer.py::TokenizerProtocol`:

```python
class TokenizerProtocol(Protocol):
    vocab_size: int
    eos_token_id: int | None

    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
```

Existem dois tokenizadores:

- `ByteTokenizer`: simples, sem dependências, útil para testes.
- `GPT2BPETokenizer`: usa `tiktoken` com vocabulário GPT-2.

O `ByteTokenizer` converte texto para bytes UTF-8. Ele tem vocabulário `257`: os valores `0..255` representam bytes, e `256` é usado como EOS, isto é, fim de texto.

O `GPT2BPETokenizer` usa BPE, o tipo de tokenização usado pelo GPT-2 original. Ele é mais realista, mas depende de `tiktoken`.

## Dataset autoregressivo

O dataset principal é `src/srp_gpt2/data/dataset.py::ParquetTextDataset`.

Ele faz três tarefas:

1. Carrega textos de um dataset Hugging Face ou Parquet.
2. Tokeniza todos os textos.
3. Cria janelas de tamanho `block_size + 1`.

Para cada janela, ele retorna:

```python
x = chunk[:-1]
y = chunk[1:]
```

Se `chunk` é:

```text
[10, 20, 30, 40, 50]
```

e `block_size=4`, então:

```text
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
```

O modelo vê `10` e tenta prever `20`; vê `10, 20` e tenta prever `30`; e assim por diante.

Shapes retornados:

- `x`: `[T]`
- `y`: `[T]`

Depois que o `DataLoader` junta vários exemplos:

- `x`: `[B, T]`
- `y`: `[B, T]`

## Stride

`stride` controla o deslocamento entre uma janela e a próxima.

Com `block_size=8` e `stride=8`, as janelas não se sobrepõem. Com `stride=4`, elas se sobrepõem pela metade. Um stride menor cria mais exemplos, mas repete mais tokens entre janelas.

## Relação com o treino

O `Trainer` espera batches neste formato:

```text
x: [B, T]
y: [B, T]
```

O modelo transforma `x` em logits:

```text
logits: [B, T, vocab]
```

A loss compara esses logits com `y`. Esse é o coração do treino autoregressivo.
