# Análise de Radiografias de Tórax e Detecção de Efusão

Este projeto realiza a extração, pré-processamento e análise exploratória de dados do **NIH Chest X-ray Dataset** (um grande conjunto de dados de radiografias de tórax com 14 rótulos de doenças). O foco principal é preparar um subconjunto do dataset para a tarefa de **classificação binária** (presença de **Efusão** vs. **Sem achado**) usando apenas radiografias na posição **PA** (Posteroanterior). Finalmente, um modelo simples de Deep Learning baseado em Transfer Learning (MobileNetV2) é construído, treinado e avaliado neste subconjunto.

## Pré-requisitos

*   Python 3.8+
*   Jupyter Notebook ou JupyterLab
*   As seguintes bibliotecas Python (serão instaladas via `pip install`):
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `Pillow`
    *   `matplotlib`
    *   `seaborn`
    *   `tensorflow`

## Configuração do Dataset

1.  Faça o download do **NIH Chest X-ray Dataset** a partir do Kaggle: [https://www.kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)
2.  O download incluirá vários arquivos `.zip` contendo as imagens (`images_001.zip` a `images_012.zip`) e um arquivo CSV (`Data_Entry_2017.csv`).
3.  Crie uma pasta chamada `archive` na raiz deste projeto.
4.  Copie o arquivo `Data_Entry_2017.csv` diretamente para a pasta `archive/`.
5.  **Importante:** Extraia *todo* o conteúdo dos arquivos `images_XXX.zip` para subpastas dentro de `archive/`, mantendo a estrutura original. Por exemplo, o conteúdo de `images_001.zip` deve ser extraído para `archive/images_001/`, o conteúdo de `images_002.zip` para `archive/images_002/`, e assim por diante. Certifique-se de que as imagens `.png` estejam nas subpastas `images` dentro dessas pastas (ex: `archive/images_001/images/00000001_000.png`).

## Estrutura do Projeto

```
.
├── README.md
├── archive/              # Pasta para o dataset original (Data_Entry_2017.csv e imagens extraídas)
│   ├── Data_Entry_2017.csv
│   ├── images_001/
│   │   └── images/
│   │       └── ... (arquivos .png)
│   ├── images_002/
│   │   └── images/
│   │       └── ... (arquivos .png)
│   └── ... (outras pastas de imagem)
├── data/                 # Pasta gerada pelos notebooks para dados processados e imagens organizadas
│   ├── data_entry.csv
│   ├── data_dictionary.csv
│   ├── train/            # Imagens para treino
│   │   ├── AP/
│   │   │   ├── 0/        # Classe 'No Finding'
│   │   │   └── 1/        # Classe 'Effusion'
│   │   └── PA/
│   │       ├── 0/        # Classe 'No Finding'
│   │       └── 1/        # Classe 'Effusion'
│   ├── validation/       # Imagens para validação
│   │   ├── AP/
│   │   │   ├── 0/
│   │   │   └── 1/
│   │   └── PA/
│   │       ├── 0/
│   │       └── 1/
│   └── test/             # Imagens para teste
│       ├── AP/
│       │   ├── 0/
│   │   │   └── 1/
│   │   └── PA/
│   │       ├── 0/
│   │       └── 1/
│   └── best_model_pa_effusion.keras # Modelo treinado salvo
├── data_extraction.ipynb # Notebook para extração e balanceamento de dados
├── copy_images.ipynb   # Notebook para redimensionamento, split e cópia de imagens
├── desc_stats.ipynb    # Notebook para análise descritiva do dataset original
└── modelo.ipynb        # Notebook para construir, treinar e avaliar o modelo
```

## Como usar

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/Helena190/Analise_Imagem_Radiologicaica
    cd Analise_Imagem_Radiologicaica
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv .venv
    # No Linux/macOS
    source .venv/bin/activate
    # No Windows
    .venv\Scripts\activate
    ```

3.  **Instale as bibliotecas necessárias:**
    Abra cada um dos notebooks (`data_extraction.ipynb`, `copy_images.ipynb`, `desc_stats.ipynb`, `modelo.ipynb`) no Jupyter/JupyterLab e execute a primeira célula `pip install`. Isso garantirá que todas as dependências sejam instaladas no seu ambiente virtual.

4.  **Configure o Dataset:** Siga as instruções na seção "Configuração do Dataset" acima para baixar e organizar os dados originais na pasta `archive/`.

5.  **Execute os Notebooks em Ordem:** É fundamental executar os notebooks na seguinte ordem, rodando *todas as células* em cada um:

    *   `data_extraction.ipynb`: Este notebook lê o CSV original, filtra e balanceia os dados para a tarefa de classificação de Efusão (vs. Sem achado) e salva o resultado em `data/data_entry.csv`. Ele também gera um dicionário de dados.
    *   `copy_images.ipynb`: Este notebook lê `data/data_entry.csv`, localiza as imagens correspondentes na pasta `archive/`, redimensiona para 256x256 pixels, divide o dataset em conjuntos de treino, validação e teste, e copia as imagens para a estrutura de pastas dentro de `data/`. **Este passo pode levar bastante tempo**, dependendo da velocidade do seu disco e CPU, pois envolve processar e copiar milhares de imagens.
    *   `desc_stats.ipynb`: (Opcional, mas recomendado para entender os dados brutos) Execute este notebook para ver análises descritivas e visualizações do dataset original completo.
    *   `modelo.ipynb`: Este notebook carrega os datasets de imagens preparados por `copy_images.ipynb`, define a arquitetura do modelo (MobileNetV2 para Transfer Learning), compila, treina e avalia o modelo para a detecção de Efusão (PA View).

## Detalhes dos Notebooks

*   `data_extraction.ipynb`: Realiza a limpeza inicial, seleciona as colunas relevantes, filtra as entradas que contêm 'Effusion' e amosta entradas 'No Finding' para criar um dataset balanceado. Mapeia os rótulos e posições de visualização para valores numéricos (0 ou 1).
*   `copy_images.ipynb`: Implementa a lógica para encontrar os caminhos completos das imagens listadas no CSV processado, abre cada imagem, redimensiona usando PIL, e salva no formato PNG em subdiretórios organizados por split (train/validation/test), posição (PA/AP) e rótulo (0/1).
*   `desc_stats.ipynb`: Utiliza `matplotlib` e `seaborn` para gerar gráficos e tabelas que mostram a distribuição de rótulos, posições de visualização, gênero e idade no dataset original, fornecendo insights sobre a composição dos dados brutos.
*   `modelo.ipynb`: Define um modelo Sequential do Keras utilizando a base do MobileNetV2 com pesos pré-treinados ('imagenet'), remove a camada de classificação original e adiciona uma camada densa com ativação sigmoid para a classificação binária. O modelo base é congelado (não treinável). O modelo é compilado com BinaryCrossentropy e o otimizador Adam. O treinamento inclui callbacks para Early Stopping e salvamento do melhor modelo com base na perda de validação. A avaliação final é feita no conjunto de teste, gerando um relatório de classificação e matriz de confusão.

## Resultados Esperados

Após a execução bem-sucedida dos notebooks, você terá:

*   A pasta `data/` contendo o CSV processado, o dicionário de dados e os subdiretórios com as imagens redimensionadas e organizadas.
*   Um arquivo `best_model_pa_effusion.keras` na pasta `data/` contendo os pesos do modelo treinado.
*   Nos notebooks, saídas mostrando estatísticas descritivas, informações de carregamento de dados, resumo do modelo, progresso do treinamento e resultados de avaliação (perda, acurácia, relatório de classificação, matriz de confusão).

## Observações

*   O redimensionamento e a cópia das imagens são feitos em CPU no notebook `copy_images.ipynb`.  Este notebook, juntamente com o de treino do modelo ()`modelo.ipynb`) podem levar um tempo considerável.
