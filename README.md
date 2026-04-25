# SRP GPT-2 PyTorch

Implementação didática e extensível de um Transformer decoder-only estilo GPT-2, organizada pelo **Single Responsibility Principle (SRP)**.

O projeto inclui:

- Arquitetura GPT-2 Small: `12` blocos, `12` heads, embedding `768`, contexto `1024`, vocab GPT-2 `50257`.
- Camadas separadas por responsabilidade: embedding, atenção causal, MLP, bloco Transformer, modelo GPT.
- Treinamento com AdamW, warmup + cosine decay, gradient accumulation, AMP opcional, gradient clipping e checkpoints.
- Inferência com `temperature`, `top_k`, `top_p` e parada por EOS.
- Tokenizador GPT-2 BPE via `tiktoken` para a didática principal e tokenizador `byte` apenas para smoke/debug.
- CLI, exemplos e testes.

> Observação: esta implementação cria um modelo **no nível arquitetural GPT-2 Small**. Treinar do zero com qualidade comparável ao GPT-2 real exige corpus massivo e várias GPUs. Para validar localmente, use a configuração tiny. Para entender tokenização GPT, use o modo GPT-2 BPE; o tokenizer byte é só ferramenta técnica de teste.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,tokenizers,hf]"
```

Sem `tiktoken`, ainda é possível rodar testes rápidos com o tokenizador byte, mas a
aplicação didática de raio-x exige `.[tokenizers]` para não ensinar uma tokenização
diferente da do GPT:

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

## Treino rápido com Parquet local

Crie um Parquet pequeno para smoke test:

```bash
mkdir -p data
python - <<'PY'
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.Table.from_pylist([
    {"text": "O rato roeu a roupa do rei de Roma."},
    {"text": "Transformers aprendem padrões autoregressivos."},
])
pq.write_table(table, "data/tiny.parquet", compression="zstd")
PY
```

Treine um modelo tiny em CPU:

```bash
python examples/train_tiny.py --parquet data/tiny.parquet --out-dir checkpoints/tiny
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

## Raio-x interativo desktop

Para abrir a janela interativa, use este launcher:

```bash
python examples/xray_desktop.py --mode train --device auto --tokenizer gpt2 --text-file data/tiny.txt
```

O app detecta `cuda`, depois `mps`/Metal, e só usa CPU se não houver acelerador.
Ele tem `play/pause`, execução passo a passo, slider de velocidade, treino ao vivo,
inferência ao vivo, pipeline visual do Transformer, grafo simplificado e atenção causal.
O padrão é `--tokenizer gpt2`, que mostra BPE/subword tokens como pedaços de palavra,
espaço + palavra e pontuação. O modo `--tokenizer byte-debug` existe só para smoke e
é rotulado como não representativo da tokenização GPT.

Para abrir direto em inferência:

```bash
python examples/xray_desktop.py \
  --mode generate \
  --device auto \
  --tokenizer gpt2 \
  --checkpoint checkpoints/tiny_xray/last.pt \
  --prompt "O rato"
```

Os scripts `examples/train_tiny_xray.py` e `examples/generate_tiny_xray.py` são
versões CLI para gerar logs/relatórios. Para janela desktop, use `examples/xray_desktop.py`.

## Raio-x em CLI

Se quiser gerar relatórios e checkpoints no terminal, use os scripts separados.
O modo `smoke` só valida que o pipeline roda; para uma demonstração legível, use
`overfit`:

```bash
python examples/train_tiny_xray.py --mode overfit --tokenizer gpt2 --text-file data/tiny.txt
```

Esse modo mostra tokenização, pares `x -> y`, loss, perplexity, grad norm, top
tokens prováveis, amostras geradas e atenção causal do último token. Ele grava:

- `checkpoints/tiny_xray/last.pt`
- `checkpoints/tiny_xray/xray/events.jsonl`
- `checkpoints/tiny_xray/xray/report.md`

Depois gere texto com rastreamento token por token:

```bash
python examples/generate_tiny_xray.py \
  --tokenizer gpt2 \
  --checkpoint checkpoints/tiny_xray/last.pt \
  --prompt "O rato"
```

Por padrão a geração didática usa `--strategy greedy`, para mostrar o caminho mais
provável aprendido. Use `--strategy sample` quando quiser demonstrar aleatoriedade,
temperature, top-k e top-p.

O rastreamento fica em `checkpoints/tiny_xray/xray/generation_trace.md`.

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

Fontes citadas no dataset card:

- Project Gutenberg, acessado via Gutendex API: https://www.gutenberg.org/ e https://gutendex.com/
- FineWeb2 da Hugging Face, filtrado para português/pt-BR: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2

## Treino no servidor

Treine direto do dataset Parquet no Hugging Face:

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
