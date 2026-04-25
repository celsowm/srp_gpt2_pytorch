# 04 - Atenção causal

Atenção é o mecanismo que permite que cada token consulte outros tokens do contexto. No GPT-2, essa consulta é causal: uma posição só pode olhar para ela mesma e para posições anteriores.

O código principal está em `src/srp_gpt2/model/attention.py::CausalSelfAttention`.

## Entrada e saída

A atenção recebe representações internas:

```text
x: [B, T, C]
```

E devolve o mesmo formato:

```text
y: [B, T, C]
```

Isso permite encaixar a atenção dentro de um bloco Transformer com conexão residual.

## Q, K e V

A primeira camada linear cria três versões de `x`:

```python
qkv = self.qkv_projection(x)
q, k, v = qkv.split(channels, dim=2)
```

Intuição:

- `Q` ou query: o que cada posição está procurando.
- `K` ou key: o que cada posição oferece como chave de busca.
- `V` ou value: a informação que será combinada.

Shapes antes de separar heads:

```text
q: [B, T, C]
k: [B, T, C]
v: [B, T, C]
```

## Múltiplas heads

Em vez de fazer uma única atenção grande, o modelo divide `C` em várias heads.

Com `n_embd=768` e `n_head=12`:

```text
head_dim = 768 / 12 = 64
```

Depois de `CausalSelfAttention._split_heads`, os shapes ficam:

```text
q: [B, n_head, T, head_dim]
k: [B, n_head, T, head_dim]
v: [B, n_head, T, head_dim]
```

Cada head aprende um tipo diferente de relação entre tokens.

## Máscara causal

Sem máscara, a posição 2 poderia olhar para a posição 5. Isso quebraria o treino autoregressivo, porque o modelo veria o futuro.

A máscara causal tem a forma de uma matriz triangular:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

O valor `1` indica atenção permitida. O valor `0` indica posição futura bloqueada.

No PyTorch 2.x, o projeto usa:

```python
F.scaled_dot_product_attention(..., is_causal=True)
```

Isso delega a implementação para uma rotina otimizada. Se essa função não existir, o projeto usa `_manual_attention`, que aplica a máscara explicitamente.

## Produto escalar escalado

A atenção compara `q` com `k`:

```text
scores = q @ k.T
```

Depois escala por `sqrt(head_dim)`, aplica máscara, `softmax` e combina com `v`.

Resultado por head:

```text
[B, n_head, T, head_dim]
```

Depois as heads são juntadas de volta:

```text
[B, T, C]
```

Por fim, `out_projection` mistura a informação das heads e retorna a saída final da atenção.

## Resumo mental

Atenção causal responde à pergunta:

> Para cada posição do texto, quais posições anteriores são mais úteis para prever o próximo token?

Essa é a peça central que diferencia um Transformer de uma rede que olha apenas para janelas fixas ou estados recorrentes.
