# SRP GPT-2 PyTorch

Implementação didática e modular de um Transformer decoder-only estilo GPT-2, organizada pelo **Single Responsibility Principle (SRP)**.

O projeto agora é **100% autossuficiente**, incluindo um motor de tokenização BPE (Byte-Pair Encoding) escrito do zero em Python puro, eliminando dependências externas complexas para o fluxo didático principal.

O projeto inclui:

- **BPE "Hand-Made"**: Algoritmo de treinamento incremental e encoding via heap ($O(N \log N)$) implementado nativamente (em `src/srp_gpt2/data/bpe.py`).
- Arquitetura GPT-2 Small: `12` blocos, `12` heads, embedding `768`, contexto `1024`, vocab ajustável.
- Camadas separadas por responsabilidade: embedding, atenção causal, MLP, bloco Transformer, modelo GPT.
- Treinamento com AdamW, warmup + cosine decay, gradient accumulation e checkpoints.
- Inferência com `temperature`, `top_k`, `top_p` e visualização via Xray.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,hf]"
```

> Nota: Não é mais necessário instalar `sentencepiece` ou `tiktoken` para usar o tokenizador customizado recomendado.

## Treinando seu próprio Tokenizador (Recomendado)

O projeto agora usa um tokenizador BPE didático otimizado para Português. Treine-o no seu dataset local:

```bash
python scripts/train_tokenizer.py \
  --input "dataset_livros_ptbr/*.txt" \
  --output data/tokenizer/ptbr_32k \
  --vocab_size 32000
```

Para usar no treino, o CLI já usa o atalho `ptbr` por padrão:

```bash
srp-gpt2 train \
  --config configs/tiny.yaml \
  --hf-dataset hf_dataset_smoke \
  --out-dir checkpoints/tiny
```

## Estrutura

```text
src/srp_gpt2/
  config.py                  # dataclasses de configuração
  cli.py                     # comandos train/generate/param-count
  model/
    attention.py             # atenção causal multi-head
    ...
  data/
    tokenizer.py             # SentencePiece, GPT-2 BPE e byte tokenizer
    dataset.py               # dataset autoregressivo de texto
  ...
scripts/
  train_tokenizer.py         # Script para treinar vocabulário customizado
```

## Treinando seu próprio Tokenizador (Recomendado)

Para máxima eficiência em português, treine o tokenizador no seu dataset local:

```bash
python scripts/train_tokenizer.py \
  --input "dataset_livros_ptbr/*.txt" \
  --output data/tokenizer/ptbr_32k \
  --vocab_size 32000
```

Para usar no treino:

```bash
srp-gpt2 train \
  --config configs/gpt2_small.yaml \
  --tokenizer data/tokenizer/ptbr_32k.model \
  ...
```

## Raio-x interativo desktop

Para abrir a janela interativa com um tokenizador customizado:

```bash
python examples/xray_desktop.py \
  --mode train \
  --tokenizer data/tokenizer/ptbr_32k.model \
  --text-file data/tiny.txt
```

O app detecta `cuda`, depois `mps`/Metal, e só usa CPU se não houver acelerador.

## Treino GPT-2 Small com Parquet no Hugging Face

```bash
srp-gpt2 train \
  --config configs/gpt2_small.yaml \
  --hf-dataset celsowm/srp-gpt2-ptbr-corpus \
  --tokenizer gpt2 \
  --out-dir checkpoints/gpt2-small \
  --device cuda
```

## Dataset

O corpus padrão é público no Hugging Face:

- `celsowm/srp-gpt2-ptbr-corpus`: https://huggingface.co/datasets/celsowm/srp-gpt2-ptbr-corpus

## SRP aplicado

Cada módulo tem uma responsabilidade principal:

- `attention.py`: calcular atenção causal multi-head.
- `feed_forward.py`: aplicar a MLP do bloco.
- `block.py`: compor norm + attention + MLP + resíduos.
- `gpt.py`: compor embeddings, blocos e head LM.
- `optimizer.py`: definir grupos com e sem weight decay.
- `scheduler.py`: calcular LR por step.
- `trainer.py`: orquestrar o treinamento, sem conhecer detalhes internos das camadas.
- `generator.py`: executar geração autoregressiva, sem treinar nem salvar checkpoints.

## Notas técnicas

- A atenção usa `torch.nn.functional.scaled_dot_product_attention` quando disponível via PyTorch 2.x.
- Os pesos de input embedding e `lm_head` são compartilhados, como em modelos GPT.
- O `lm_head` pode calcular loss internamente quando `targets` é passado.
- O tokenizador byte é útil para testes e demonstrações, mas para uso real prefira SentencePiece treinado no seu corpus.
