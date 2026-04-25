# 05 - Bloco Transformer: A Linha de Montagem

Agora juntamos tudo: embeddings, atenção causal, MLP e normalização. O Bloco Transformer é como uma estação em uma linha de montagem, onde a informação é refinada passo a passo.

Os arquivos principais são:
- `src/srp_gpt2/model/block.py`: A estrutura do bloco individual.
- `src/srp_gpt2/model/gpt.py`: O "esqueleto" que empilha os blocos.
- `src/srp_gpt2/model/loss.py`: Como o modelo sabe que errou.

## O Bloco: "Pre-Norm" e Conexões Residuais

A arquitetura do GPT-2 segue o padrão:
```python
x = x + self.attention(self.ln_1(x))
x = x + self.feed_forward(self.ln_2(x))
```

### 1. A Rodovia de Informação (Resíduos)
Imagine que o vetor `x` é uma **rodovia principal**. As camadas de Atenção e MLP são **vias laterais** (desvios) onde a informação é processada e depois "injetada" de volta na rodovia principal via soma (`x = x + ...`).
- **Didática**: Isso permite que o gradiente flua sem obstáculos durante o treino, evitando que o modelo "esqueça" o que aprendeu nas camadas iniciais. Se uma camada for inútil, o modelo pode simplesmente aprender a zerar sua contribuição e a informação passa direto pela rodovia.

### 2. LayerNorm (O Estabilizador)
O `ln_1` e `ln_2` (Layer Normalization) garantem que os números não fiquem grandes demais nem pequenos demais. No GPT-2, usamos **Pre-Norm**: normalizamos a informação **antes** dela entrar na atenção ou na MLP. Isso torna o treinamento muito mais estável em modelos profundos.

## O Modelo GPT: A Pilha de Conhecimento

O `GPTLanguageModel` monta a arquitetura completa:
1.  **Entrada**: IDs de tokens viram vetores ricos + GPS de posição.
2.  **Blocos**: Empilhamos vários blocos (ex: 12 no GPT-2 Small). Cada bloco refina a compreensão (Gramática -> Lógica -> Estilo).
3.  **LayerNorm Final**: Um último ajuste de escala antes da saída.
4.  **Cabeça Final (LM Head)**: Transforma o vetor abstrato de volta em um score para cada palavra do vocabulário (logits).

## Weight Tying: "O Caminho de Volta"

O projeto compartilha os pesos entre o `token_embedding` e o `lm_head`.
- **Intuição**: Se o embedding aprendeu que o vetor [0.1, 0.5...] significa "cachorro", faz sentido que a cabeça de saída use esse mesmo vetor para identificar quando o modelo quer dizer "cachorro". 
- **Benefício**: Reduz drasticamente a memória necessária e melhora a generalização.

## A Função de Perda (Loss): O Professor

Em `src/srp_gpt2/model/loss.py`, usamos a **Cross Entropy**.
- O modelo tenta prever o token $T+1$ usando apenas as informações até a posição $T$.
- O "erro" (loss) é a diferença entre a probabilidade que o modelo deu para a palavra correta e o que realmente aconteceu.

## Inicialização e Flexibilidade

- **Inicialização**: Em `src/srp_gpt2/model/init.py`, usamos uma técnica especial para reduzir o desvio padrão nas camadas residuais, evitando que o modelo "exploda" no início do treino.
- **Crop Context**: O modelo pode ter seu contexto (block_size) reduzido após o treino para economizar memória em dispositivos menores, sem precisar de re-treinamento.
