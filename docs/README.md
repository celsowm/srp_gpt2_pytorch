# Construindo um GPT-2 com PyTorch

Esta série explica, passo a passo, como este projeto constrói um modelo estilo GPT-2 usando PyTorch. O foco é didático: entender o fluxo completo do código real, sem tentar cobrir toda a matemática de Transformers em profundidade.

O caminho principal é:

```text
texto -> tokens -> exemplos autoregressivos -> embeddings -> atenção causal
      -> blocos Transformer -> logits -> loss -> treino -> geração
```

## Pré-requisitos

Antes de começar, é útil saber:

- Python básico.
- Noções iniciais de PyTorch: `torch.Tensor`, `nn.Module`, `DataLoader`.
- Ideia geral de treino supervisionado: entrada, alvo, loss, backward e optimizer.

Você não precisa dominar Transformers antes de ler. Cada arquivo introduz os conceitos necessários.

## Ordem recomendada

1. [Visão geral](01-visao-geral.md)
2. [Configuração, tokenização e dataset](02-configuracao-tokenizacao-dataset.md)
3. [Embeddings e MLP](03-embeddings-e-mlp.md)
4. [Atenção causal](04-atencao-causal.md)
5. [Bloco Transformer e modelo GPT](05-bloco-transformer-e-modelo.md)
6. [Treinamento](06-treinamento.md)
7. [Geração](07-geracao.md)
8. [Prática](08-pratica.md)

## Como ler o código junto

Sempre que aparecer uma referência como `src/srp_gpt2/model/gpt.py::GPTLanguageModel`, leia como:

- arquivo: `src/srp_gpt2/model/gpt.py`
- classe ou função: `GPTLanguageModel`

A documentação evita copiar arquivos inteiros. Em vez disso, ela explica o papel de cada parte e mostra pequenos trechos quando ajudam a entender a mecânica.

## Nota importante

Este projeto implementa uma arquitetura compatível com a ideia do GPT-2 Small: modelo decoder-only, atenção causal, embeddings aprendidos, blocos Transformer e cabeça de linguagem. Treinar um modelo com qualidade parecida com o GPT-2 original exige corpus massivo, muitas horas de GPU e ajuste cuidadoso. Para estudo local, use a configuração tiny e o tokenizador byte.
