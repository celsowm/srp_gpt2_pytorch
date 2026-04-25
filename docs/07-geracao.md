# 07 - Geração: A Arte de Contar Histórias

Após o treino, o modelo se torna um mestre da probabilidade. Gerar texto é um processo iterativo onde o modelo "adivinha" a próxima palavra e depois a lê para adivinhar a seguinte.

Os arquivos principais são:
- `src/srp_gpt2/inference/generator.py`: O laço infinito de geração.
- `src/srp_gpt2/inference/sampler.py`: O cérebro que escolhe entre as opções.

## O Ciclo de Geração (Autoregressão)

O `TextGenerator` funciona como um escritor que nunca volta atrás:
1.  **Contexto**: Recebe o texto inicial (Prompt).
2.  **Previsão**: O modelo gera scores (logits) para todas as 32.000 palavras do dicionário.
3.  **Amostragem**: O `Sampler` escolhe uma dessas palavras baseado nos scores.
4.  **Feedback**: A palavra escolhida é colada no final do texto original.
5.  **Repetição**: O processo recomeça, agora com um contexto um pouco maior.

## Controlando a "Personalidade" do Modelo

O `Sampler` possui vários parâmetros que alteram como ele escolhe a próxima palavra:

### 1. Temperature (O "Clima")
- **Baixa (< 0.7)**: O modelo fica conservador e previsível. Ele sempre escolhe as palavras mais óbvias.
- **Alta (> 1.0)**: O modelo fica criativo e "caótico". Ele começa a considerar opções menos prováveis, o que pode gerar textos surpreendentes ou sem sentido.

### 2. Top-K e Top-P (Filtros de Qualidade)
- **Top-K**: O modelo só pode escolher entre as `K` melhores opções (ex: top 50). Isso joga fora a "cauda longa" de palavras totalmente sem nexo.
- **Top-P (Nucleus)**: O modelo soma as probabilidades das melhores palavras até chegar em `P` (ex: 95%). Se ele estiver muito confiante, escolherá entre 2 ou 3 palavras. Se estiver confuso, abrirá o leque para 20 ou 30.

### 3. Repetition Penalty (Anti-Gagueira)
Modelos de linguagem às vezes entram em loops infinitos ("eu gosto de chocolate e chocolate e chocolate..."). Esta penalidade reduz o score de palavras que já apareceram recentemente, forçando o modelo a ser mais variado.

## Parada por EOS (Fim de Jogo)

O tokenizador possui um token especial chamado **EOS (End Of Sequence)**. Quando o modelo escolhe esse token, o gerador entende que a história acabou e interrompe o processo automaticamente.
