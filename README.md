# Análise de Radiografias de Tórax e Detecção de Efusão

Este projeto realiza a extração, pré-processamento e análise exploratória de dados do **NIH Chest X-ray Dataset** (um grande conjunto de dados de radiografias de tórax com 14 rótulos de doenças). O foco principal é preparar um subconjunto do dataset para a tarefa de **classificação binária** (presença de **Efusão** vs. **Sem achado**) usando apenas radiografias na posição **PA** (Posteroanterior). Finalmente, um modelo simples de Deep Learning baseado em Transfer Learning (MobileNetV2) é construído, treinado e avaliado neste subconjunto.

## Como funciona

Este projeto agora inclui uma **interface web interativa** construída com Flask, permitindo o upload de imagens de radiografias de tórax para classificação em tempo real. Além disso, o projeto é estruturado em pipelines modulares que podem ser executados de forma independente ou em conjunto. O ponto de entrada principal para os pipelines de dados e modelo é o arquivo [`src/main.py`](src/main.py), que utiliza `argparse` para controlar a execução das diferentes etapas:

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
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                      # Aplicação web Flask para classificação de imagens.
├── data/                       # Diretório para dados (Gerado).
│   ├── data_dictionary.csv     # Dicionário de dados processado .
│   ├── data_entry.csv          # Arquivo de entrada original do dataset.
│   ├── source_dictionary.csv   # Dicionário de dados original.
│   ├── train/                  # Subconjunto de dados para treinamento.
│   ├── validation/             # Subconjunto de dados para validação.
│   └── test/                   # Subconjunto de dados para teste.
├── logs/                       # Diretório para arquivos de log (Gerado).
├── models/                     # Diretório para modelos treinados (Gerado).
├── reports/                    # Diretório para relatórios e gráficos (Gerado).
├── src/
│   ├── __init__.py
│   ├── main.py                 # Ponto de entrada principal e orquestrador dos pipelines.
│   ├── analysis/               # Módulos para análise de dados.
│   │   ├── __init__.py
│   │   └── descriptive_statistics.py
│   ├── data/                   # Módulos para carregamento e processamento de dados.
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_processor.py
│   │   ├── data_saver.py
│   │   └── image_processor.py
│   ├── models/                 # Módulos para construção e avaliação de modelos.
│   │   ├── __init__.py
│   │   ├── data_loader_tf.py
│   │   ├── model_builder.py
│   │   ├── model_evaluator.py
│   │   └── model_trainer.py
│   └── utils/                  # Módulos de utilidades.
│       ├── __init__.py
│       ├── constants.py        # Módulo para constantes do projeto.
│       └── logger.py           # Módulo para configuração de logging.
├── templates/                  # Diretório para templates HTML da aplicação Flask.
│   └── index.html              # Template principal da interface web.
└── uploads/                    # Diretório para imagens carregadas pela interface web (Gerado).
```

## Configuração e Instalação

Para configurar e instalar o ambiente do projeto, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/Helena190/Analise_Imagem_Radiologicaica
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

O projeto pode ser executado de duas formas principais:

1.  **Execução dos Pipelines de Dados e Modelo (via `src/main.py`)**:
    O arquivo [`src/main.py`](src/main.py) permite executar os pipelines de dados, análise e modelo de forma modular.

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

2.  **Execução da Aplicação Web Flask (`app.py`)**:
    Para iniciar a interface web e realizar classificações de imagens, execute o arquivo `app.py`. Certifique-se de que o modelo de Deep Learning já foi treinado e salvo no diretório `models/`, pois a aplicação tentará carregá-lo na inicialização.

    ```bash
    python app.py
    ```
    Após a execução, a aplicação estará disponível em `http://127.0.0.1:5000/` (ou outra porta, dependendo da configuração).

