# 06 - Treinamento

Treinar o GPT significa ajustar os pesos para reduzir o erro de previsão do próximo token.

Os arquivos principais são:

- `src/srp_gpt2/training/optimizer.py`
- `src/srp_gpt2/training/scheduler.py`
- `src/srp_gpt2/training/checkpoint.py`
- `src/srp_gpt2/training/trainer.py`
- `src/srp_gpt2/cli.py`

## Entrada do loop de treino

O `Trainer` recebe:

- modelo.
- optimizer.
- scheduler.
- `train_loader`.
- `val_loader` opcional.
- configurações.
- diretório de saída.
- device.

Cada batch do `DataLoader` tem:

```text
x: [B, T]
y: [B, T]
```

O modelo retorna:

```text
logits: [B, T, vocab]
loss: escalar
```

## AdamW

`src/srp_gpt2/training/optimizer.py::build_adamw` cria o optimizer.

O projeto separa parâmetros em dois grupos:

- com weight decay: matrizes de pesos principais.
- sem weight decay: biases, LayerNorm e embeddings.

Weight decay é uma regularização que desestimula pesos muito grandes. Nem todo parâmetro deve recebê-lo; por isso o agrupamento é explícito.

O teste `tests/test_model.py::test_optimizer_covers_trainable_parameters_once` verifica que todos os parâmetros treináveis entram no optimizer uma única vez.

## Scheduler: warmup + cosine

`src/srp_gpt2/training/scheduler.py::WarmupCosineScheduler` controla o learning rate.

O comportamento é:

```text
fase 1: warmup
  learning rate sobe gradualmente

fase 2: cosine decay
  learning rate desce suavemente até min_learning_rate
```

Warmup ajuda no início do treino, quando os pesos ainda estão aleatórios e updates muito grandes podem desestabilizar o modelo.

## Gradient accumulation

`Trainer._training_step` divide o batch em microbatches:

```python
chunks_x = x.chunk(self.train_config.gradient_accumulation_steps)
chunks_y = y.chunk(self.train_config.gradient_accumulation_steps)
```

Isso permite simular um batch maior sem colocar tudo de uma vez na GPU.

Exemplo:

```text
batch real: 16
gradient_accumulation_steps: 4
microbatch: 4
```

O modelo processa 4 exemplos por vez, acumula gradientes e só depois faz `optimizer.step()`.

## AMP

AMP significa automatic mixed precision.

No projeto, se `amp=True` e o device for CUDA, o treino usa:

```python
torch.autocast(device_type="cuda", dtype=torch.bfloat16)
```

Isso pode reduzir memória e acelerar treino em GPUs compatíveis.

## Gradient clipping

Antes do passo do optimizer, o projeto pode aplicar:

```python
torch.nn.utils.clip_grad_norm_(...)
```

Gradient clipping limita gradientes muito grandes. Isso ajuda a evitar instabilidade no treino.

## Checkpoints

`src/srp_gpt2/training/checkpoint.py::CheckpointManager` salva:

- pesos do modelo.
- estado do optimizer.
- estado do scheduler.
- estado do treino.
- configurações.

O `Trainer` salva:

- `last.pt`: checkpoint mais recente.
- `best.pt`: melhor checkpoint quando existe validação e a loss melhora.

## CLI de treino

`src/srp_gpt2/cli.py::train_command` conecta tudo:

```text
config YAML
  -> tokenizer
  -> dataset
  -> DataLoader
  -> modelo
  -> optimizer
  -> scheduler
  -> Trainer.fit()
```

O comando principal é:

```bash
srp-gpt2 train \
  --config configs/gpt2_small.yaml \
  --hf-dataset celsowm/srp-gpt2-ptbr-corpus \
  --tokenizer gpt2 \
  --out-dir checkpoints/gpt2-small \
  --device cuda
```

Para estudo local, prefira a prática tiny do último capítulo.
