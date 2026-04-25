# 04 - Atenção Causal: O Mecanismo de Consulta

A Atenção é o "superpoder" do Transformer. Ela permite que cada token decida quais outros tokens no contexto são relevantes para entender o presente e prever o futuro. No GPT-2, essa atenção é **Causal** (ou "mascarada"): um token só pode olhar para o passado.

O código principal está em `src/srp_gpt2/model/attention.py::CausalSelfAttention`.

## A Analogia da Biblioteca: Q, K e V

Para entender como a atenção funciona matematicamente, imagine uma biblioteca:

1.  **Query (Q)**: É o seu **cartão de busca**. Ele contém o que você está procurando no momento (ex: "quem é o sujeito desta frase?").
2.  **Key (K)**: É a **etiqueta na lombada** dos livros. Cada livro na estante (contexto) oferece uma chave para ser encontrado.
3.  **Value (V)**: É o **conteúdo do livro**. Se o seu cartão (Q) der um "match" com uma etiqueta (K), você extrai a informação (V) daquele livro.

O modelo calcula a similaridade entre `Q` e `K` para decidir quanto de cada `V` ele deve absorver.

## Múltiplas Heads: Os "Especialistas"

Em vez de uma única atenção gigante, dividimos o vetor em 12 "heads" (cabeças).
- **Por que?** Imagine 12 especialistas lendo a mesma frase.
- O especialista 1 foca em **Gramática**.
- O especialista 2 foca em **Entidades** (nomes de pessoas/lugares).
- O especialista 3 foca em **Pontuação**.
Ao final, combinamos a opinião de todos para formar a saída final.

## A Máscara Causal: "Proibido dar Spoiler"

Um modelo autoregressivo como o GPT-2 deve prever o próximo token sem saber qual é. Se ele pudesse olhar para a frente durante o treino, ele simplesmente "copiaria" a resposta.

A **Máscara Causal** é um triângulo de zeros e "menos infinito" (ou booleanos) que bloqueia a visão do futuro:
- Na posição 5, o modelo vê as posições 0, 1, 2, 3, 4 e 5.
- As posições 6, 7, 8... são invisíveis (mascaradas).

## Resumo Visual do Cálculo

1.  **Projeção**: Criamos Q, K, V a partir da entrada.
2.  **Scores**: Multiplicamos $Q \times K^T$ (quem combina com quem?).
3.  **Escalonamento**: Dividimos pela raiz quadrada da dimensão (para manter os números estáveis).
4.  **Máscara**: Zeramos as conexões com o futuro.
5.  **Softmax**: Transformamos os scores em probabilidades que somam 100%.
6.  **Mistura**: Multiplicamos essas probabilidades por V.

Atenção causal responde à pergunta:
> "Dado o que escrevi até agora, para onde devo olhar para decidir o que escrever a seguir?"
