# 03 - Embeddings e MLP

Depois que o texto vira IDs, o modelo precisa transformar esses números em vetores. Essa etapa é feita pelos embeddings.

Os arquivos principais são:

- `src/srp_gpt2/model/embeddings.py`
- `src/srp_gpt2/model/feed_forward.py`

## Embeddings de token

Um token ID sozinho é apenas um número inteiro. Por exemplo, o token `42` não tem significado matemático especial.

`src/srp_gpt2/model/embeddings.py::TokenPositionEmbeddings` usa `nn.Embedding` para mapear cada ID para um vetor aprendido:

```text
input_ids: [B, T]
token_emb: [B, T, C]
```

Onde:

- `B`: batch size.
- `T`: quantidade de tokens no contexto.
- `C`: dimensão do embedding, isto é, `n_embd`.

Se `n_embd=768`, cada token vira um vetor de 768 números.

## Embeddings de posição

Atenção por si só não sabe a ordem dos tokens. Para informar posição, o GPT-2 usa embeddings posicionais aprendidos.

O projeto cria posições assim:

```python
positions = torch.arange(0, time, dtype=torch.long, device=input_ids.device)
```

Depois soma:

```text
token_emb + pos_emb
```

Shapes:

```text
token_emb: [B, T, C]
pos_emb:   [1, T, C]
saida:     [B, T, C]
```

O `pos_emb` tem batch `1`, mas o PyTorch faz broadcasting para aplicar as mesmas posições a todos os itens do lote.

## Limite de contexto

`TokenPositionEmbeddings.forward` valida se `T <= block_size`. Isso é importante porque a tabela de posições foi criada com tamanho máximo fixo.

Se `block_size=1024`, o modelo só tem embeddings posicionais para posições `0..1023`.

## MLP do bloco

Depois da atenção, cada bloco Transformer usa uma MLP posição a posição. No projeto, ela fica em `src/srp_gpt2/model/feed_forward.py::FeedForward`.

A estrutura é:

```text
Linear(C -> 4C)
GELU
Linear(4C -> C)
Dropout
```

Em código, a ideia aparece como:

```python
nn.Linear(config.n_embd, 4 * config.n_embd)
nn.GELU(approximate="tanh")
nn.Linear(4 * config.n_embd, config.n_embd)
```

Essa MLP não mistura posições diferentes. Ela transforma cada posição individualmente:

```text
entrada: [B, T, C]
saida:   [B, T, C]
```

## Por que expandir para 4C?

A expansão `C -> 4C -> C` dá mais capacidade para o modelo criar combinações não lineares. A atenção mistura informação entre tokens; a MLP refina a representação de cada posição.

No GPT-2, atenção e MLP trabalham juntas dentro de cada bloco.
