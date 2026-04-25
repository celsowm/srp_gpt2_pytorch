# 08 - Prática

Este roteiro mostra como executar o projeto em escala pequena para validar o fluxo completo.

O objetivo não é treinar um GPT-2 bom. O objetivo é ver o pipeline funcionando:

```text
criar dados -> treinar tiny -> salvar checkpoint -> gerar texto
```

## Instalação

Crie e ative um ambiente virtual. No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,hf,tokenizers]"
```

Se quiser instalar o mínimo para o exemplo byte:

```powershell
pip install -e ".[dev,hf]"
```

## Criar um Parquet pequeno

O exemplo `examples/train_tiny.py` espera um arquivo Parquet com coluna `text`.

Crie um arquivo pequeno em `data/tiny.parquet`:

```powershell
python -c "import pyarrow as pa, pyarrow.parquet as pq; table = pa.Table.from_pylist([{'text': 'O rato roeu a roupa do rei de Roma.'}, {'text': 'Transformers aprendem padrões autoregressivos.'}, {'text': 'Atenção causal impede olhar para o futuro.'}]); pq.write_table(table, 'data/tiny.parquet', compression='zstd')"
```

## Treinar o modelo tiny

Execute:

```powershell
python examples/train_tiny.py --parquet data/tiny.parquet --out-dir checkpoints/tiny
```

Esse script usa:

- `ByteTokenizer`
- `ModelConfig` pequeno
- `ParquetTextDataset`
- `DataLoader`
- `GPTLanguageModel`
- `AdamW`
- `WarmupCosineScheduler`
- `Trainer`

O modelo tiny é pequeno para rodar rápido. Ele serve como smoke test do pipeline.

## Gerar texto

Depois do treino:

```powershell
python examples/generate.py `
  --checkpoint checkpoints/tiny/last.pt `
  --tokenizer byte `
  --prompt "Transformers usam atenção causal" `
  --max-new-tokens 30 `
  --temperature 0.6 `
  --top-k 20 `
  --top-p 0.85 `
  --repetition-penalty 1.1
```

Como o dataset é minúsculo, o texto pode sair repetitivo ou estranho. Isso é esperado.

## Contar parâmetros

Para contar parâmetros do GPT-2 Small:

```powershell
srp-gpt2 param-count --config configs/gpt2_small.yaml
```

Esse comando instancia `GPTLanguageModel` usando a configuração YAML e imprime a contagem de parâmetros treináveis.

## Treinar com dataset Hugging Face

Para treino realista com tokenizador GPT-2:

```powershell
srp-gpt2 train `
  --config configs/gpt2_small.yaml `
  --hf-dataset celsowm/srp-gpt2-ptbr-corpus `
  --tokenizer gpt2 `
  --out-dir checkpoints/gpt2-small `
  --device cuda
```

Esse treino exige GPU e tempo. Para uma máquina comum, use configs menores ou o exemplo tiny.

## Gerar com checkpoint da CLI

```powershell
srp-gpt2 generate `
  --checkpoint checkpoints/gpt2-small/last.pt `
  --tokenizer gpt2 `
  --prompt "Era uma vez" `
  --max-new-tokens 120 `
  --temperature 0.8 `
  --top-k 50 `
  --top-p 0.95
```

## Checklist de entendimento

Ao terminar, você deve conseguir explicar:

- Por que o dataset retorna `x` e `y` deslocados.
- Por que a entrada do modelo tem shape `[B, T]`.
- Por que os embeddings produzem `[B, T, C]`.
- Por que a atenção precisa de máscara causal.
- Por que os logits finais têm shape `[B, T, vocab]`.
- Como a geração escolhe um token por vez.

Se esses pontos ficaram claros, você já entendeu a espinha dorsal de um GPT-2 em PyTorch.
