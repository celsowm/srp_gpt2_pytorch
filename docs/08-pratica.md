# 08 - Prática: Do Texto ao Modelo

Neste roteiro final, vamos executar o pipeline completo de "ponta a ponta" usando apenas as ferramentas que construímos. O objetivo é validar que você agora possui um sistema de IA funcional e transparente.

## O Pipeline SRP

O fluxo que vamos seguir é:
1.  **Dataset**: Preparar textos em português.
2.  **Tokenização**: Treinar o BPE "Hand-Made".
3.  **Treino**: Ensinar a GPT-2 a prever os tokens.
4.  **Inferência**: Gerar texto e ver o Raio-X.

---

## 1. Preparação (Tokenizador Customizado)

Em vez de usar modelos prontos da OpenAI, vamos criar o nosso. Use o visualizador para treinar um vocabulário de 2000 tokens no dataset de livros fornecido:

```bash
python examples/train_tokenizer_xray.py
```
*(No app, selecione `dataset_livros_ptbr/*.txt` e clique em Start Training).*

Isso criará os arquivos em `data/tokenizer/`.

---

## 2. Treinamento da GPT-2 Tiny

Para validar o código sem precisar de um supercomputador, vamos usar a configuração `tiny`. Ela é pequena o suficiente para treinar na CPU em poucos minutos.

Primeiro, prepare o dataset de fumaça (smoke test):
```bash
# Isso cria um arquivo Parquet pequeno para teste
python -c "import pyarrow as pa, pyarrow.parquet as pq; table = pa.Table.from_pylist([{'text': 'O rato roeu a roupa do rei de Roma.'}] * 100); pq.write_table(table, 'hf_dataset_smoke/train.parquet')"
```

Agora, dispare o treino:
```bash
python -m srp_gpt2.cli train \
  --config configs/tiny.yaml \
  --hf-dataset hf_dataset_smoke \
  --tokenizer ptbr \
  --out-dir checkpoints/tiny
```

---

## 3. Geração e Inspeção (Raio-X)

Agora que o modelo "aprendeu" o padrão do rato e do rei, vamos ver como ele pensa. Use a ferramenta de Raio-X interativo:

```bash
python examples/transformer_desktop_xray.py \
  --checkpoint checkpoints/tiny/last.pt \
  --tokenizer ptbr
```

**O que observar:**
- Como o texto é quebrado em peças de BPE (subwords).
- Quais palavras o modelo considera mais prováveis para continuar a frase.
- O efeito da **Temperature** na barra lateral.

---

## 4. Checklist Final de Maestria

Ao concluir este projeto, você deve ser capaz de apontar no código:
- [ ] Onde a **Atenção** impede que o modelo "veja o futuro" (`attention.py`).
- [ ] Onde o **BPE** aprende que "qu" + "e" = "que" (`bpe.py`).
- [ ] Onde a **Rodovia de Resíduos** permite o fluxo de gradiente (`block.py`).
- [ ] Onde o **Amostrador** joga fora palavras improváveis (`sampler.py`).

**Parabéns!** Você não apenas "usou" uma IA; você construiu uma do zero, peça por peça, seguindo os princípios de Single Responsibility.
