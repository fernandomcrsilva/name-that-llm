# Name That LLM: qual modelo escreveu este texto?

Esta é a minha participação na competição [Name That LLM](https://www.kaggle.com/competitions/name-that-llm), do Kaggle. A tarefa é ler um texto e dizer qual de 18 LLMs o escreveu: Claude, GPT, Gemini e outros. A métrica é a acurácia balanceada, que dá o mesmo peso a cada um dos 18 modelos. É uma competição de código, então a submissão é um notebook que roda sem internet, em no máximo 9 horas.

**Resultado:** 0,481 no placar público, na posição 3.147 de 4.969 equipes (em 07/11/2026). O primeiro lugar tinha 0,747. Não é uma posição de destaque. O que vale neste repositório é o processo: uma validação que passou a prever o placar, um erro caro que o placar revelou e a medida de quanto cada ideia ajudou.

## Os dados e a validação

O treino tem 2.699 textos. São 150 prompts, e cada um foi respondido pelos 18 modelos. O teste usa prompts que não aparecem no treino.

Isso define como validar. Se textos do mesmo prompt caem no treino e na validação ao mesmo tempo, o classificador aprende o tema do prompt, e a nota sobe sem significar nada. Por isso, todas as validações deste repositório separam os textos por prompt, com `GroupKFold` sobre o `prompt_id`.

## O caminho, versão por versão

| Versão | O que mudou | Validação | Placar público |
|---|---|---|---|
| Baseline | TF-IDF de caracteres e palavras com regressão logística | 0,342 | 0,345 |
| v2 | Embeddings do Qwen3-1.7B, com truques que dependem dos grupos de prompt | 0,605 | 0,349 |
| v3 | Os mesmos modelos, sem nenhum truque de grupo | 0,406 | 0,380 |
| v4 | Treino com textos públicos extras, mistura dos modelos e balanceamento | 0,511 | **0,481** |

### O erro que custou mais caro: a v2

No treino, cada prompt tem exatamente um texto de cada modelo. A v2 apostou nessa estrutura:

1. reconstruí os grupos de prompt no teste, juntando os textos de tema parecido;
2. centralizei os embeddings dentro de cada grupo, para tirar o tema e deixar o estilo;
3. impus a regra "cada modelo aparece uma vez por grupo", resolvendo a atribuição com o algoritmo húngaro.

A validação foi a 0,605. O placar ficou em 0,349, praticamente igual ao baseline.

A validação estava medindo a coisa errada. Ela reproduzia a estrutura do treino, em que cada grupo tem os 18 modelos, e por isso só podia confirmar a aposta. O placar mostrou que o teste não segue essa estrutura, ou que os grupos que eu reconstruí não batiam com os dele. Nenhuma validação feita só com o treino tinha como detectar isso, porque ela herdava a mesma suposição.

Na v3 tirei todos os truques de grupo e deixei o classificador olhar só para o texto de cada passagem. A validação caiu para 0,406, mas a distância entre validação e placar caiu de 0,26 para 0,03. A partir daí, cada ganho na validação passou a valer no placar.

### O que funcionou: mais textos de treino

Com 2.699 textos para 18 classes, o limite é a quantidade de exemplos. A comunidade da competição publicou textos gerados pelos mesmos modelos, e o [Name That LLM Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) junta 212 mil deles, vindos de 60 conjuntos públicos. O pool já vem sem os textos que repetem prompts do treino, então usá-lo não vaza informação para a validação.

Os textos extras entram no treino com peso 0,5, e a validação continua usando só textos oficiais. Com um SVM linear sobre TF-IDF, o efeito do pool foi:

| Textos extras | Validação |
|---|---|
| nenhum | 0,318 |
| 40 mil | 0,444 |
| 40 mil, sem os marcados como gerados por intermediário | 0,442 |
| 100 mil | 0,478 |

Os textos gerados por intermediário não atrapalharam, então ficaram. Nos embeddings do Qwen3-1.7B com regressão logística, 40 mil textos extras levaram a validação de 0,374 para 0,440.

### Mistura e balanceamento

A v4 mistura os dois classificadores (0,3 embeddings e 0,7 SVM) e chega a 0,511 na validação.

Por fim, um balanceamento global suave (Sinkhorn) puxa a proporção prevista de cada modelo para perto de uma meta, que é o meio do caminho entre a proporção prevista e a uniforme. O ajuste entra só em parte. Na validação, o ganho foi praticamente nulo: de 0,5108 para 0,5112. Ele fica na submissão como uma aposta barata, não como um resultado.

O placar da v4 foi 0,481, de novo a 0,03 da validação.

## Como reproduzir

O caminho mais curto é o [`kaggle/v4/submission_kernel.py`](kaggle/v4/submission_kernel.py), que faz a submissão inteira num arquivo só, dentro do Kaggle. Para refazer as validações localmente:

```bash
kaggle competitions download name-that-llm -p data && unzip data/name-that-llm.zip -d data
kaggle datasets download crtproduction/name-that-llm-open-pool -p extra/pool --unzip
python -c "import pandas as pd; p = pd.read_parquet('extra/pool/ntl_open_pool.parquet'); p.sample(frac=1, random_state=0).groupby('model').head(40000 // 18)[['id', 'text', 'model']].to_csv('extra/pool_sample40k.csv', index=False)"

python src/baseline.py                                                     # 0,342
python src/extra_cv.py --pool 100000 --salvar runs/svm100k_oof.npy        # 0,478
python src/embed.py --model Qwen/Qwen3-1.7B --csv data/train.csv --out runs/emb-qwen3-1.7b-train.npy
python src/embed.py --model Qwen/Qwen3-1.7B --csv data/train.csv --layers 0.2,0.3,0.4 --out runs/emb-qwen3-1.7b-shallow-train.npy
python src/embed.py --model Qwen/Qwen3-1.7B --csv extra/pool_sample40k.csv --layers 0.3,0.5 --out runs/emb-q17-pool40k.npy
python src/emb_cv.py                                                       # 0,374, 0,440 e a mistura 0,511
```

| Arquivo | Função |
|---|---|
| [`src/baseline.py`](src/baseline.py) | TF-IDF e regressão logística, validados por prompt |
| [`src/extra_cv.py`](src/extra_cv.py) | Quanto os textos extras ajudam o SVM |
| [`src/embed.py`](src/embed.py) | Embeddings de camadas intermediárias de um LLM congelado |
| [`src/emb_cv.py`](src/emb_cv.py) | Embeddings com e sem textos extras, mistura e balanceamento |
| [`src/balance.py`](src/balance.py) | Balanceamento global suave (Sinkhorn) |
| [`src/assign.py`](src/assign.py) | Os truques de grupo da v2: reconstrução dos grupos e atribuição húngara |
| [`src/train_cls.py`](src/train_cls.py) | Experimento com um classificador Qwen3 com LoRA, que não entrou na submissão |
| [`kaggle/`](kaggle/) | Os notebooks de cada submissão |

## Dados, créditos e licenças

- Os dados da competição seguem as regras do Kaggle e não estão neste repositório.
- O [Name That LLM Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool), de crtproduction, reúne 60 conjuntos públicos sob CC BY 4.0, CC BY-SA 4.0 e CC0. A lista completa de fontes e licenças está no `sources.csv` do próprio conjunto. Agradeço a quem gerou e publicou esses textos. Eles também não estão neste repositório: os comandos acima os baixam.
- O [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) está sob Apache 2.0.
- O código deste repositório está sob [MIT](LICENSE).
