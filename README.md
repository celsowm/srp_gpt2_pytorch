# SRP GPT-2 PyTorch

Implementação didática e extensível de um Transformer decoder-only estilo GPT-2, organizada pelo **Single Responsibility Principle (SRP)**.

O projeto inclui:

- Arquitetura GPT-2 Small: `12` blocos, `12` heads, embedding `768`, contexto `1024`, vocab GPT-2 `50257`.
- Camadas separadas por responsabilidade: embedding, atenção causal, MLP, bloco Transformer, modelo GPT.
- Treinamento com AdamW, warmup + cosine decay, gradient accumulation, AMP opcional, gradient clipping e checkpoints.
- Inferência com `temperature`, `top_k`, `top_p` e parada por EOS.
- Tokenizador GPT-2 via `tiktoken` opcional e tokenizador `byte` sem dependências para testes rápidos.
- CLI, exemplos e testes.

> Observação: esta implementação cria um modelo **no nível arquitetural GPT-2 Small**. Treinar do zero com qualidade comparável ao GPT-2 real exige corpus massivo e várias GPUs. Para validar localmente, use a configuração tiny ou byte-tokenizer.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,tokenizers]"
```

Sem `tiktoken`, ainda é possível treinar e gerar com o tokenizador byte:

```bash
pip install -e ".[dev]"
```

## Estrutura

```text
src/srp_gpt2/
  config.py                  # dataclasses de configuração
  cli.py                     # comandos train/generate/param-count
  model/
    attention.py             # atenção causal multi-head
    block.py                 # bloco Transformer
    embeddings.py            # token + positional embeddings
    feed_forward.py          # MLP GPT-2
    gpt.py                   # composição do modelo
    init.py                  # inicialização de pesos
    loss.py                  # cross entropy
  data/
    tokenizer.py             # GPT-2 BPE e byte tokenizer
    dataset.py               # dataset autoregressivo de texto
  training/
    optimizer.py             # grupos AdamW
    scheduler.py             # warmup + cosine
    checkpoint.py            # save/load
    trainer.py               # loop de treino
  inference/
    sampler.py               # temperature/top-k/top-p
    generator.py             # geração autoregressiva
```

## Parâmetros GPT-2 Small

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

Isso resulta em aproximadamente **124M parâmetros**, equivalente ao GPT-2 Small.

## Treino rápido com tokenizador byte

Crie um arquivo pequeno:

```bash
mkdir -p data
cat > data/tiny.txt <<'TXT'
O rato roeu a roupa do rei de Roma.
Transformers aprendem padrões autoregressivos.
TXT
```

Treine um modelo tiny em CPU:

```bash
python examples/train_tiny.py --text data/tiny.txt --out-dir checkpoints/tiny
```

Gere texto:

```bash
python examples/generate.py \
  --checkpoint checkpoints/tiny/last.pt \
  --tokenizer byte \
  --prompt "Transformers usam atenção causal" \
  --max-new-tokens 30 \
  --temperature 0.6 \
  --top-k 20 \
  --top-p 0.85 \
  --repetition-penalty 1.1
```

## Treino GPT-2 Small

```bash
srp-gpt2 train \
  --config configs/gpt2_small.yaml \
  --train-text data/train.txt \
  --val-text data/val.txt \
  --tokenizer gpt2 \
  --out-dir checkpoints/gpt2-small \
  --device cuda
```

## Treino com dataset Parquet no Hugging Face

Instale as dependências do Hugging Face:

```bash
pip install -e ".[dev,tokenizers,hf]"
```

Autentique no servidor se o dataset for privado:

```bash
hf auth login
```

Treine sem gerar arquivos `.txt` locais:

```bash
srp-gpt2 train \
  --config configs/gpt2_server_h100_resume_3060.yaml \
  --hf-dataset celsowm/srp-gpt2-ptbr-corpus \
  --tokenizer gpt2 \
  --out-dir checkpoints/gpt2-h100 \
  --device cuda \
  --gpu-index 7
```

Para continuar um checkpoint feito com a config da RTX 3060, use uma config com o
mesmo `block_size=256`, como `configs/gpt2_server_h100_resume_3060.yaml`. A config
`configs/gpt2_server_h100.yaml` usa `block_size=1024` e deve ser usada para treino
novo, sem retomar checkpoint de `block_size=256`.

Caso queira apenas construir o modelo e contar parâmetros:

```bash
srp-gpt2 param-count --config configs/gpt2_small.yaml
```

## Geração

```bash
srp-gpt2 generate \
  --checkpoint checkpoints/gpt2-small/last.pt \
  --tokenizer gpt2 \
  --prompt "Era uma vez" \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95
```

## Testes

```bash
pytest
```

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
- O tokenizador byte é útil para testes e demonstrações, mas GPT-2 real usa BPE.
