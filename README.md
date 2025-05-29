# Análise de Radiografias de Tórax e Detecção de Efusão

Este projeto realiza a extração, pré-processamento e análise exploratória de dados do **NIH Chest X-ray Dataset** (um grande conjunto de dados de radiografias de tórax com 14 rótulos de doenças). O foco principal é preparar um subconjunto do dataset para a tarefa de **classificação binária** (presença de **Efusão** vs. **Sem achado**) usando apenas radiografias na posição **PA** (Posteroanterior). Finalmente, um modelo simples de Deep Learning baseado em Transfer Learning (MobileNetV2) é construído, treinado e avaliado neste subconjunto.

## Como funciona

O projeto é estruturado em pipelines modulares que podem ser executados de forma independente ou em conjunto. O ponto de entrada principal é o arquivo [`src/main.py`](src/main.py), que utiliza `argparse` para controlar a execução das diferentes etapas:

*   **Pipeline de Dados (`--data`)**: Responsável pela extração, pré-processamento e salvamento dos dados originais e processados, incluindo o processamento das imagens.
    *   `DataLoader`: Carrega os dados originais.
    *   `DataProcessor`: Processa o DataFrame, filtrando e preparando os dados.
    *   `DataSaver`: Salva os dados processados e dicionários de dados.
    *   `ImageProcessor`: Realiza o processamento das imagens.
*   **Pipeline de Análise (`--analyze`)**: Executa a análise estatística descritiva dos dados.
    *   `DescriptiveStatistics`: Realiza análises e gera relatórios.
*   **Pipeline de Modelo (`--model`)**: Abrange o treinamento e a avaliação do modelo de aprendizado de máquina.
    *   `TFDataLoader`: Carrega os datasets no formato TensorFlow.
    *   `ModelBuilder`: Constrói o modelo de Transfer Learning (MobileNetV2).
    *   `ModelTrainer`: Treina o modelo.
    *   `ModelEvaluator`: Avalia o desempenho do modelo.

## Estrutura

A estrutura de diretórios do projeto é organizada da seguinte forma:

```
.
├── .gitignore                  # Arquivo de configuração do Git para ignorar arquivos e diretórios.
├── README.md                   # Este arquivo de documentação do projeto.
├── requirements.txt            # Lista de dependências Python do projeto.
├── data/                       # Diretório para dados.
│   ├── data_dictionary.csv     # Dicionário de dados processado (Gerado).
│   ├── data_entry.csv          # Arquivo de entrada original do dataset.
│   ├── source_dictionary.csv   # Dicionário de dados original (Gerado).
│   ├── train/                  # Subconjunto de dados para treinamento (Gerado).
│   │   ├── AP/                 # Imagens na posição AP.
│   │   │   ├── 0/              # Imagens sem efusão.
│   │   │   └── 1/              # Imagens com efusão.
│   │   └── PA/                 # Imagens na posição PA.
│   │       ├── 0/              # Imagens sem efusão.
│   │       └── 1/              # Imagens com efusão.
│   ├── validation/             # Subconjunto de dados para validação (Gerado).
│   │   ├── PA/                 # Imagens na posição PA.
│   │       ├── 0/              # Imagens sem efusão.
│   │       └── 1/              # Imagens com efusão.
├── logs/                       # Diretório para arquivos de log.
│   └── app.log                 # Log principal da aplicação (Gerado).
├── models/                     # Diretório para modelos treinados.
│   └── best_model_pa_effusion.keras # Modelo de Deep Learning treinado (Gerado).
├── reports/                    # Diretório para relatórios e gráficos gerados.
│   ├── age_distribution_by_view_position.png # Gráfico de distribuição de idade por posição de visualização (Gerado).
│   ├── average_age_per_finding.png # Gráfico de idade média por achado (Gerado).
│   ├── findings_frequency.png  # Gráfico de frequência de achados (Gerado).
│   ├── gender_distribution_per_finding.png # Gráfico de distribuição de gênero por achado (Gerado).
│   ├── patient_gender_distribution.png # Gráfico de distribuição de gênero do paciente (Gerado).
│   └── view_positions_by_gender.png # Gráfico de posições de visualização por gênero (Gerado).
└── src/                        # Código fonte do projeto.
    ├── __init__.py             # Inicializa o pacote Python.
    ├── main.py                 # Ponto de entrada principal e orquestrador dos pipelines.
    ├── analysis/               # Módulos para análise de dados.
    │   ├── __init__.py         # Inicializa o pacote.
    │   └── descriptive_statistics.py # Módulo para estatísticas descritivas.
    ├── data/                   # Módulos para carregamento e processamento de dados.
    │   ├── __init__.py         # Inicializa o pacote.
    │   ├── data_loader.py      # Módulo para carregar dados.
    │   ├── data_processor.py   # Módulo para processar dados.
    │   ├── data_saver.py       # Módulo para salvar dados.
    │   └── image_processor.py  # Módulo para processar imagens.
    ├── models/                 # Módulos para construção e avaliação de modelos.
    │   ├── __init__.py         # Inicializa o pacote.
    │   ├── data_loader_tf.py   # Módulo para carregar dados para TensorFlow.
    │   ├── model_builder.py    # Módulo para construir o modelo.
    │   ├── model_evaluator.py  # Módulo para avaliar o modelo.
    │   └── model_trainer.py    # Módulo para treinar o modelo.
    └── utils/                  # Módulos de utilidades.
        ├── __init__.py         # Inicializa o pacote.
        ├── constants.py        # Módulo para constantes do projeto.
        └── logger.py           # Módulo para configuração de logging.
```

## Configuração e Instalação

Para configurar e instalar o ambiente do projeto, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd efusao-raiox
    ```
2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Linux/macOS
    # venv\Scripts\activate   # No Windows
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## Execução

O projeto pode ser executado a partir do arquivo [`src/main.py`](src/main.py) utilizando diferentes flags para ativar os pipelines desejados.

**Comandos de Execução:**

*   **Executar o pipeline de dados:**
    ```bash
    python -m src.main --data
    ```
*   **Executar o pipeline de análise:**
    ```bash
    python -m src.main --analyze
    ```
*   **Executar o pipeline do modelo:**
    ```bash
    python -m src.main --model
    ```
*   **Executar todos os pipelines (dados, análise e modelo) em sequência:**
    ```bash
    python -m src.main --all
    ```
*   **Visualizar as opções de ajuda:**
    ```bash

    python -m src.main --help
    ```

**Configuração de Logging:**

Você pode ajustar o nível de detalhe dos logs usando a flag `--log_level`. Os níveis disponíveis são `DEBUG`, `INFO`, `WARNING`, `ERROR` e `CRITICAL`. Por padrão, o nível é `INFO`.

*   **Exemplo de execução com nível de log DEBUG:**
    ```bash
    python -m src.main --all --log_level DEBUG
    ```

Os logs são gravados no arquivo [`logs/app.log`](logs/app.log).
