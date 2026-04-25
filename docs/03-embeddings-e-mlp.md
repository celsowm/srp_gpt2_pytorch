# 03 - Embeddings e MLP: Transformando Números em Significado

Depois que o texto vira IDs pelo tokenizador, o modelo precisa dar "cor" e "contexto" a esses números. Essa etapa transforma IDs frios em vetores ricos em significado.

## Embeddings de Token: O "Espaço de Significados"

Um token ID como `42` é apenas um índice. O modelo não sabe se `42` é "cachorro" ou "computador". 

A camada `src/srp_gpt2/model/embeddings.py::TokenPositionEmbeddings` utiliza uma tabela de busca (`nn.Embedding`) que funciona como um **dicionário de coordenadas**:
- Cada palavra ganha um endereço em um espaço de 768 dimensões (`n_embd`).
- **Didática**: Imagine que palavras similares (como "gato" e "cachorro") terminam com coordenadas próximas nesse espaço, enquanto "gato" e "algoritmo" moram em bairros distantes.

## Embeddings de Posição: O "GPS" do Modelo

O mecanismo de Atenção (que veremos a seguir) é "agnóstico à ordem". Para ele, a frase "O rato comeu o queijo" e "queijo o comeu rato O" parecem idênticas.

Para resolver isso, somamos um **Vetor de Posição**:
- Cada posição (0, 1, 2...) tem seu próprio vetor único.
- Ao somar `token_emb + pos_emb`, "carimbamos" em cada palavra a informação de onde ela está na frase.
- **Importante**: O GPT-2 usa posições aprendidas, o que significa que o modelo descobre sozinho a melhor forma de representar o conceito de "primeira palavra" ou "última palavra".

## MLP (Feed Forward): O Token "Pensando Sozinho"

Após a Atenção misturar informações de diferentes tokens, cada token passa pela MLP em `src/srp_gpt2/model/feed_forward.py`.

### A Estrutura C -> 4C -> C
A MLP funciona como uma **Refinaria de Informação**:
1. **Expansão (C -> 4C)**: Projetamos o vetor de 768 para 3072 dimensões. Isso dá ao modelo um "espaço de rascunho" maior para fazer cálculos complexos.
2. **Ativação (GELU)**: Uma função não-linear que decide quais informações são importantes (funciona como um filtro inteligente).
3. **Projeção (4C -> C)**: Comprime a informação de volta para 768 dimensões, mantendo apenas o que foi refinado.

**Diferença Crucial**: 
- A **Atenção** é sobre **comunicação** entre tokens.
- A **MLP** é sobre **processamento individual**. Cada token "reflete" sobre a informação que recebeu da atenção sem olhar para os vizinhos.
