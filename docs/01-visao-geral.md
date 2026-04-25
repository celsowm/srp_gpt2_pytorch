# 01 - Visão geral

Um GPT-2 é um modelo de linguagem autoregressivo. Isso significa que ele aprende a prever o próximo token a partir dos tokens anteriores.

Exemplo simples:

```text
entrada: O gato subiu no
alvo:    gato subiu no muro
```

O modelo recebe uma sequência deslocada e tenta acertar o próximo token em cada posição. Durante a geração, ele repete esse processo: prevê um token, adiciona esse token ao contexto e prevê o próximo.

## O que significa decoder-only

Transformers podem ter encoder, decoder ou os dois. O GPT-2 usa apenas a parte decoder.

Na prática, isso quer dizer:

- O modelo lê tokens da esquerda para a direita.
- Cada posição só pode olhar para posições anteriores.
- A máscara causal impede que o token atual veja tokens futuros.
- A saída final é uma distribuição de probabilidade sobre o vocabulário.

No projeto, essa ideia aparece principalmente em:

- `src/srp_gpt2/model/attention.py::CausalSelfAttention`
- `src/srp_gpt2/model/block.py::TransformerBlock`
- `src/srp_gpt2/model/gpt.py::GPTLanguageModel`

## Fluxo do projeto

O fluxo completo fica assim:

```text
1. O texto é convertido em IDs de tokens.
2. O dataset cria pares x/y para prever o próximo token.
3. O modelo transforma IDs em embeddings.
4. Cada bloco Transformer aplica atenção causal e MLP.
5. A cabeça final produz logits para cada token do vocabulário.
6. A loss compara logits com os alvos.
7. O treino ajusta os pesos com AdamW.
8. A geração usa os logits para escolher novos tokens.
```

Shapes importantes:

- `[B, T]`: lote de IDs de tokens. `B` é batch size, `T` é tamanho da sequência.
- `[B, T, C]`: representações internas. `C` é a dimensão do embedding.
- `[B, T, vocab]`: logits finais, um score para cada token do vocabulário.

## Mapa dos módulos

O código é separado por responsabilidade:

```text
src/srp_gpt2/
  config.py                  # configurações do modelo, treino e dados
  data/
    tokenizer.py             # tokenizadores byte e GPT-2 BPE
    dataset.py               # dataset autoregressivo
  model/
    embeddings.py            # token + posição
    attention.py             # atenção causal multi-head
    feed_forward.py          # MLP do bloco
    block.py                 # bloco Transformer
    gpt.py                   # modelo completo
    loss.py                  # cross entropy
    init.py                  # inicialização dos pesos
  training/
    optimizer.py             # AdamW com grupos de weight decay
    scheduler.py             # warmup + cosine decay
    checkpoint.py            # salvar e carregar checkpoints
    trainer.py               # loop de treino
  inference/
    sampler.py               # temperature, top-k, top-p
    generator.py             # geração autoregressiva
```

Essa organização ajuda o estudo: você pode entender o GPT-2 como uma sequência de peças pequenas.

## GPT-2 Small neste projeto

A configuração `configs/gpt2_small.yaml` define:

```yaml
model:
  vocab_size: 50257
  block_size: 1024
  n_layer: 12
  n_head: 12
  n_embd: 768
  dropout: 0.1
  bias: true
```

Isso cria um modelo no nível arquitetural do GPT-2 Small, com cerca de 124M parâmetros. A arquitetura é comparável, mas a qualidade final depende do volume de dados e do custo de treino.
