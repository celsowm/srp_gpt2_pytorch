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

- `vocab_size`: quantidade de tokens possíveis. **Nota:** Se usar um tokenizador customizado, o `vocab_size` será ajustado automaticamente para bater com o vocabulário treinado.
- `block_size`: tamanho máximo do contexto.
- `n_layer`: número de blocos Transformer.
- `n_head`: número de cabeças de atenção.
- `n_embd`: tamanho do vetor interno de cada token.

## Tokenização

Modelos de linguagem não recebem texto cru. Eles recebem IDs inteiros. Diferente de outros projetos que usam bibliotecas fechadas (C++), este projeto utiliza um **motor BPE (Byte-Pair Encoding) 100% nativo em Python**, localizado em `src/srp_gpt2/data/bpe.py`.

### Tokenizadores disponíveis:

1.  **`SentencePieceTokenizer`**: **(Recomendado)** Utiliza o motor BPE nativo. É a escolha didática para o projeto, permitindo ver o vocabulário "nascendo" do zero.
2.  **`ByteTokenizer`**: Simples, sem dependências. Converte texto para bytes UTF-8 (0-255). Útil para testes rápidos e fumaça.
3.  **`GPT2BPETokenizer`** (Legado): Usa o vocabulário original do GPT-2 via `tiktoken`.

### Como o BPE aprende? (O Algoritmo de Merge)

O BPE segue um processo iterativo simples, mas poderoso:

1.  **Inicialização**: Começamos com um vocabulário de caracteres individuais (a, b, c, d...).
2.  **Contagem de Pares**: O algoritmo varre o texto e conta quais dois símbolos aparecem juntos com mais frequência (ex: "q" seguido de "u").
3.  **Fusão (Merge)**: O par mais frequente é "promovido" a um novo símbolo único (ex: `q` + `u` → `qu`).
4.  **Atualização**: Todas as ocorrências de `q` e `u` adjacentes no texto original são substituídas pelo novo símbolo `qu`.
5.  **Repetição**: O processo volta ao passo 2 até atingir o `vocab_size` desejado.

**Exemplo Prático:**
Texto: `banana`
- Símbolos iniciais: `b`, `a`, `n`, `a`, `n`, `a`
- Par mais frequente: `an` (aparece 2 vezes)
- Novo símbolo: `an`
- Texto atualizado: `b`, `a`, `n`, `an`, `an` (considerando as fusões possíveis)
- Próximo merge: `an` + `a` → `ana`

Ao final de 32.000 iterações, o modelo terá aprendido que " que", " de", " para" e até palavras inteiras são unidades fundamentais do português, economizando processamento da GPT.

### Visualizando o Treino (BPE X-Ray)

Para entender como o algoritmo BPE encontra os melhores pares de caracteres para fundir, criamos uma ferramenta visual interativa:

```bash
python examples/train_tokenizer_xray.py
```

### Treinando via CLI

Para processar grandes volumes de texto (como os 128MB de livros em PT-BR fornecidos), usamos o treinamento otimizado com suporte a heap e tabelas incrementais:

```bash
python scripts/train_tokenizer.py --input "dataset_livros_ptbr/*.txt" --output data/tokenizer/ptbr_32k --vocab_size 32000
```

Isso gerará:
- `data/tokenizer/ptbr_32k.model`: O modelo em JSON legível.
- `data/tokenizer/ptbr_32k.vocab`: Lista humana dos tokens com seus scores.

Para usar este tokenizador no treino do GPT, o CLI já habilita o atalho `ptbr` por padrão:

```bash
srp-gpt2 train --config configs/tiny.yaml --tokenizer ptbr ...
```

## Dataset autoregressivo

O dataset principal é `src/srp_gpt2/data/dataset.py::ParquetTextDataset` (para uso com Hugging Face/Parquet) ou `TextFileLanguageModelDataset` (para arquivos `.txt` locais).

Ele faz três tarefas:
1. Carrega os textos.
2. Tokeniza o conteúdo.
3. Cria janelas de tamanho `block_size + 1`.

Para cada janela, ele retorna:
```python
x = chunk[:-1]
y = chunk[1:]
```
O modelo vê `x` e tenta prever `y`. Esse deslocamento de 1 token é o que define o treino autoregressivo (causal).
