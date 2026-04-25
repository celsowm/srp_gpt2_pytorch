# 05 - Bloco Transformer e modelo GPT

Agora juntamos as peças: embeddings, atenção causal, MLP, normalização, cabeça de linguagem e loss.

Os arquivos principais são:

- `src/srp_gpt2/model/block.py`
- `src/srp_gpt2/model/gpt.py`
- `src/srp_gpt2/model/loss.py`
- `src/srp_gpt2/model/init.py`

## Bloco Transformer

`src/srp_gpt2/model/block.py::TransformerBlock` representa um bloco GPT.

A estrutura é:

```python
x = x + self.attention(self.ln_1(x))
x = x + self.feed_forward(self.ln_2(x))
```

Isso mostra três ideias importantes.

Primeiro, existe `LayerNorm` antes da atenção e antes da MLP. Esse padrão é chamado de pre-norm.

Segundo, existe conexão residual. A saída da atenção é somada ao `x` original. A saída da MLP também é somada ao resultado anterior.

Terceiro, o shape se mantém:

```text
entrada: [B, T, C]
saida:   [B, T, C]
```

Manter o mesmo shape permite empilhar vários blocos.

## Modelo completo

`src/srp_gpt2/model/gpt.py::GPTLanguageModel` monta o modelo:

```text
input_ids
  -> embeddings
  -> bloco 1
  -> bloco 2
  -> ...
  -> bloco N
  -> LayerNorm final
  -> lm_head
  -> logits
```

Entrada:

```text
input_ids: [B, T]
```

Depois dos embeddings:

```text
x: [B, T, C]
```

Depois da cabeça final:

```text
logits: [B, T, vocab]
```

Cada posição recebe um vetor com um score para cada token possível.

## Weight tying

O projeto compartilha os pesos entre o embedding de tokens e a cabeça final:

```python
self.lm_head.weight = self.embeddings.token_embedding.weight
```

Isso é chamado de weight tying.

Intuição:

- Na entrada, a tabela transforma token ID em vetor.
- Na saída, a cabeça compara vetores internos com os vetores dos tokens.

Compartilhar os pesos reduz parâmetros e segue uma prática comum em modelos GPT.

O teste `tests/test_model.py::test_weight_tying` garante que os dois módulos apontam para a mesma memória.

## Loss de linguagem causal

Se `targets` for passado para `GPTLanguageModel.forward`, o modelo calcula a loss:

```python
loss = causal_lm_loss(logits, targets)
```

`src/srp_gpt2/model/loss.py::causal_lm_loss` usa cross entropy.

Shapes:

```text
logits:  [B, T, vocab]
targets: [B, T]
loss:    escalar
```

A função reorganiza os tensores para comparar cada posição com seu alvo:

```text
[B, T, vocab] -> [B*T, vocab]
[B, T]        -> [B*T]
```

## Inicialização dos pesos

`src/srp_gpt2/model/init.py::GPTWeightInitializer` inicializa `Linear` e `Embedding` com distribuição normal.

Também reduz o desvio padrão de projeções ligadas ao caminho residual. Isso ajuda a manter a escala das ativações sob controle quando muitos blocos são empilhados.

## Cortando o contexto

`GPTLanguageModel.crop_block_size` permite reduzir `block_size` depois que o modelo já existe. Isso é útil para inferência ou ajuste com contexto menor.

Ele não aumenta o contexto. A regra é:

```text
novo block_size <= block_size atual
```

Isso existe porque a tabela de posições e a máscara causal foram criadas para um tamanho máximo.
