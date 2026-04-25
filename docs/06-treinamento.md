# 06 - Treinamento: Ensinando o Modelo

Treinar o GPT-2 significa ajustar milhões de pesos internos para que o erro de previsão do próximo token seja o menor possível. É um processo de **otimização matemática** em larga escala.

Os arquivos principais são:
- `src/srp_gpt2/training/trainer.py`: O orquestrador do loop.
- `src/srp_gpt2/training/optimizer.py`: Quem decide para onde girar os botões.
- `src/srp_gpt2/training/scheduler.py`: Quem decide a velocidade da mudança.

## O Loop de Treino: Ajuste de Precisão

Imagine que o modelo tem milhões de botões de ajuste. Treinar é como ouvir um rádio chiando e girar milimetricamente cada botão até que a música (o texto) fique nítida.
1. **Forward**: O modelo faz uma previsão.
2. **Loss**: O "professor" diz o quanto ele errou.
3. **Backward**: O PyTorch calcula como cada botão contribuiu para o erro (Gradientes).
4. **Step**: O Optimizer gira os botões na direção certa.

## AdamW: O Otimizador com Memória

Usamos o **AdamW**. Ele não apenas olha para onde o erro diminui agora, mas guarda um "momentum" (uma inércia) de para onde ele estava indo. 
- **Weight Decay**: Aplicamos uma "taxa de esquecimento" em pesos grandes para evitar que o modelo decore o dataset (overfitting). O SRP GPT-2 separa inteligentemente quais pesos recebem essa taxa (matrizes) e quais não (biases e normas).

## Learning Rate: Warmup e Cosine Decay

A velocidade com que giramos os botões (Learning Rate) muda durante o treino:
1. **Warmup (Aquecimento)**: Começamos muito devagar. O modelo está instável no início, então updates bruscos podem "quebrá-lo".
2. **Cosine Decay**: Após o aquecimento, diminuímos a velocidade suavemente seguindo uma curva de cosseno. Isso ajuda o modelo a convergir para o ponto de perfeição no final do treino.

## Truques de Engenharia para GPUs Reais

- **Gradient Accumulation**: Se sua GPU é pequena, processamos o texto em "micro-batches" e acumulamos o erro antes de atualizar o modelo. É como ler 4 páginas de um livro e só depois fazer um resumo.
- **Mixed Precision (AMP)**: Usamos números de 16 bits (`bfloat16`) em vez de 32 bits. Isso dobra a velocidade e reduz o uso de memória sem perder qualidade.
- **Gradient Clipping**: Se o modelo der um "salto" muito grande no erro, nós cortamos esse salto para evitar que o treinamento exploda.
