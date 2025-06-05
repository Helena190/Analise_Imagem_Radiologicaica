import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils.logger import logger
from src.utils.constants import DATA_ENTRY_FILENAME, FINDING_LABELS_COL, VIEW_POSITION_COL, PATIENT_GENDER_COL, PATIENT_AGE_COL, IMAGE_INDEX_COL, PATIENT_ID_COL

PLOTS_DIR = 'reports'

sns.set_style('whitegrid')

class DescriptiveStatistics:
    """análise estatística descritiva e visualizações para o dataset original."""
    def __init__(self, data_entry_filename=DATA_ENTRY_FILENAME):
        from src.data.data_loader import DataLoader
        self.data_loader = DataLoader(data_entry_filename=data_entry_filename)
        self.original_df = None
        logger.info("descriptive statistics inicializado.")

    def load_data(self):
        """carrega o csv de entrada de dados original."""
        logger.info("tentando carregar dados originais para análise descritiva...")
        self.original_df = self.data_loader.load_original_data()
        if self.original_df is not None:
            logger.info(f"dataset carregado com sucesso! total de entradas: {len(self.original_df)}")
            logger.info("\nprimeiras 5 linhas do dataset:\n" + str(self.original_df.head()))
        else:
            logger.error("falha ao carregar dados originais para análise descritiva.")

    def display_general_info(self):
        """exibe informações gerais e estatísticas descritivas do dataframe."""
        if self.original_df is not None:
            logger.info("\ninformações gerais do dataframe original:")
            self.original_df.info()
            logger.info("\nestatísticas descritivas para colunas numéricas:")
            numerical_cols = [
                'Follow-up #', PATIENT_AGE_COL, 'OriginalImage[Width',
                'Height]', 'OriginalImagePixelSpacing[x', 'y]'
            ]
            existing_numerical_cols = [col for col in numerical_cols if col in self.original_df.columns]
            if existing_numerical_cols:
                logger.info(self.original_df[existing_numerical_cols].describe())
            else:
                logger.warning("nenhuma coluna numérica adequada encontrada para estatísticas descritivas.")
        else:
            logger.warning("dataframe não carregado. pulando exibição de informações gerais.")

    def analyze_finding_labels(self):
        """analisa e visualiza a distribuição de 'finding labels'."""
        if self.original_df is not None and FINDING_LABELS_COL in self.original_df.columns:
            logger.info(f"\nanalisando coluna '{FINDING_LABELS_COL}':")
            all_labels = self.original_df[FINDING_LABELS_COL].str.split('|').explode()
            label_counts = all_labels.value_counts()
            logger.info(f"contagem de ocorrências para cada achado ou condição:\n{label_counts}")

            label_counts_excluding_nofinding = label_counts.drop('No Finding', errors='ignore')
            if not label_counts_excluding_nofinding.empty:
                plt.figure(figsize=(12, 8))
                sns.barplot(x=label_counts_excluding_nofinding.index,
                            y=label_counts_excluding_nofinding.values,
                            hue=label_counts_excluding_nofinding.index,
                            palette='viridis',
                            legend=False)
                plt.title('Frequência de Achados (Excluindo "Sem Achado")', fontsize=16)
                plt.xlabel('Achado Clínico', fontsize=12)
                plt.ylabel('Número de Ocorrências', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_DIR, 'findings_frequency.png'))
                plt.close()
            else:
                logger.warning("nenhum achado além de 'no finding' para plotar ou 'no finding' é o único rótulo.")
        else:
            logger.warning(f"dataframe não carregado ou coluna '{FINDING_LABELS_COL}' não encontrada. pulando análise de 'finding labels'.")

    def analyze_categorical_columns(self):
        """analisa e visualiza 'view position' e 'patient gender'."""
        if self.original_df is not None:
            if VIEW_POSITION_COL in self.original_df.columns:
                logger.info(f"\nanalisando coluna '{VIEW_POSITION_COL}':")
                position_counts = self.original_df[VIEW_POSITION_COL].value_counts()
                logger.info(f"contagem de ocorrências para cada posição de visualização:\n{position_counts}")
                plt.figure(figsize=(6, 5))
                sns.countplot(data=self.original_df, x=VIEW_POSITION_COL, hue=VIEW_POSITION_COL, palette='pastel', legend=False)
                plt.title('Distribuição da Posição de Visualização', fontsize=14)
                plt.xlabel('Posição de Visualização', fontsize=12)
                plt.ylabel('Número de Imagens', fontsize=12)
                plt.savefig(os.path.join(PLOTS_DIR, 'view_position_distribution.png'))
                plt.close()
            else:
                logger.warning(f"coluna '{VIEW_POSITION_COL}' não encontrada. pulando sua análise.")

            if PATIENT_GENDER_COL in self.original_df.columns:
                logger.info(f"\nanalisando coluna '{PATIENT_GENDER_COL}':")
                gender_counts = self.original_df[PATIENT_GENDER_COL].value_counts()
                logger.info(f"contagem de ocorrências para cada gênero:\n{gender_counts}")
                plt.figure(figsize=(6, 5))
                sns.countplot(data=self.original_df, x=PATIENT_GENDER_COL, hue=PATIENT_GENDER_COL, palette='coolwarm', legend=False)
                plt.title('Distribuição de Gênero do Paciente', fontsize=14)
                plt.xlabel('Gênero', fontsize=12)
                plt.ylabel('Número de Imagens', fontsize=12)
                plt.savefig(os.path.join(PLOTS_DIR, 'patient_gender_distribution.png'))
                plt.close()
            else:
                logger.warning(f"coluna '{PATIENT_GENDER_COL}' não encontrada. pulando sua análise.")
        else:
            logger.warning("dataframe não carregado. pulando análise de colunas categóricas.")

    def analyze_patient_age(self):
        """analisa e visualiza a distribuição de 'patient age'."""
        if self.original_df is not None and PATIENT_AGE_COL in self.original_df.columns:
            logger.info(f"\nanalisando coluna '{PATIENT_AGE_COL}':")
            self.original_df[PATIENT_AGE_COL] = pd.to_numeric(self.original_df[PATIENT_AGE_COL], errors='coerce')
            logger.info(f"estatísticas descritivas para idade do paciente:\n{self.original_df[PATIENT_AGE_COL].describe()}")

            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.original_df, x=PATIENT_AGE_COL, bins=30, kde=True, color='skyblue')
            
            # ajusta os limites do eixo x dinamicamente ou define um máximo razoável
            min_age = self.original_df[PATIENT_AGE_COL].min()
            max_age = self.original_df[PATIENT_AGE_COL].max()
            if min_age is not None and max_age is not None:
                plt.xlim(min_age - 5 if min_age > 5 else 0, max_age + 5 if max_age < 120 else 120)
            else:
                plt.xlim(0, 100) # padrão se min/max forem none

            plt.title('Distribuição da Idade do Paciente', fontsize=16)
            plt.xlabel('Idade', fontsize=12)
            plt.ylabel('Frequência', fontsize=12)
            plt.savefig(os.path.join(PLOTS_DIR, 'patient_age_distribution.png'))
            plt.close()
        else:
            logger.warning(f"dataframe não carregado ou coluna '{PATIENT_AGE_COL}' não encontrada. pulando análise de 'patient age'.")

    def count_unique_identifiers(self):
        """conta e registra o número de imagens e pacientes únicos."""
        if self.original_df is not None:
            num_total_entries = len(self.original_df)
            num_unique_images = self.original_df[IMAGE_INDEX_COL].nunique() if IMAGE_INDEX_COL in self.original_df.columns else 0
            num_unique_patients = self.original_df[PATIENT_ID_COL].nunique() if PATIENT_ID_COL in self.original_df.columns else 0

            logger.info("\ncontando imagens e pacientes únicos:")
            logger.info(f"total de entradas no csv: {num_total_entries}")
            logger.info(f"número de nomes de arquivos de imagem únicos: {num_unique_images}")
            logger.info(f"número de ids de pacientes únicos: {num_unique_patients}")
        else:
            logger.warning("dataframe não carregado. pulando contagem de identificadores únicos.")

    def analyze_correlations(self):
        """analisa e visualiza correlações entre atributos."""
        if self.original_df is not None:
            logger.info("\n--- análise de correlação ---")

            # 1. correlação entre variáveis numéricas
            logger.info("\n1. correlação entre variáveis numéricas (pearson):")
            numerical_cols_to_check = [
                'Follow-up #', PATIENT_AGE_COL, 'OriginalImage[Width',
                'Height]', 'OriginalImagePixelSpacing[x', 'y]'
            ]
            numerical_cols = [col for col in numerical_cols_to_check if col in self.original_df.columns]

            if PATIENT_AGE_COL in self.original_df.columns:
                self.original_df[PATIENT_AGE_COL] = pd.to_numeric(self.original_df[PATIENT_AGE_COL], errors='coerce')

            if numerical_cols:
                correlation_matrix = self.original_df[numerical_cols].dropna().corr()
                logger.info(f"matriz de correlação:\n{correlation_matrix}")
                plt.figure(figsize=(len(numerical_cols)*2, len(numerical_cols)*2))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
                plt.title('matriz de correlação (numérica)', fontsize=14)
                plt.savefig(os.path.join(PLOTS_DIR, 'correlation_matrix_numerical.png'))
                plt.close()
            else:
                logger.warning("nenhuma coluna numérica adequada para calcular a matriz de correlação.")

            # 2. relação entre idade do paciente e gênero/posição de visualização
            logger.info("\n2. relação entre idade do paciente e gênero/posição de visualização:")
            if PATIENT_GENDER_COL in self.original_df.columns and PATIENT_AGE_COL in self.original_df.columns:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=self.original_df, x=PATIENT_GENDER_COL, y=PATIENT_AGE_COL, palette='coolwarm', hue=PATIENT_GENDER_COL, legend=False)
                plt.title('distribuição de idade por gênero', fontsize=14)
                plt.xlabel('gênero', fontsize=12)
                plt.ylabel('idade do paciente', fontsize=12)
                plt.savefig(os.path.join(PLOTS_DIR, 'age_distribution_by_gender.png'))
                plt.close()
            else:
                logger.warning(f"colunas '{PATIENT_GENDER_COL}' ou '{PATIENT_AGE_COL}' não encontradas para plotar idade por gênero.")
            
            if VIEW_POSITION_COL in self.original_df.columns and PATIENT_AGE_COL in self.original_df.columns:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=self.original_df, x=VIEW_POSITION_COL, y=PATIENT_AGE_COL, palette='pastel', hue=VIEW_POSITION_COL, legend=False)
                plt.title('distribuição de idade por posição de visualização', fontsize=14)
                plt.xlabel('posição de visualização', fontsize=12)
                plt.ylabel('idade do paciente', fontsize=12)
                plt.savefig(os.path.join(PLOTS_DIR, 'age_distribution_by_view_position.png'))
                plt.close()
            else:
                logger.warning(f"colunas '{VIEW_POSITION_COL}' ou '{PATIENT_AGE_COL}' não encontradas para plotar idade por posição de visualização.")
            logger.info("boxplots mostram mediana, quartis e potenciais outliers na idade para cada categoria.")

            # 3. relação entre gênero e posição de visualização
            logger.info("\n3. relação entre gênero e posição de visualização:")
            if PATIENT_GENDER_COL in self.original_df.columns and VIEW_POSITION_COL in self.original_df.columns:
                gender_position_crosstab = pd.crosstab(self.original_df[PATIENT_GENDER_COL], self.original_df[VIEW_POSITION_COL])
                logger.info(f"crosstab (gênero vs posição de visualização):\n{gender_position_crosstab}")
                plt.figure(figsize=(8, 6))
                sns.countplot(data=self.original_df, x=VIEW_POSITION_COL, hue=PATIENT_GENDER_COL, palette='viridis')
                plt.title('contagem de posições de visualização por gênero', fontsize=14)
                plt.xlabel('posição de visualização', fontsize=12)
                plt.ylabel('número de imagens', fontsize=12)
                plt.legend(title='gênero')
                plt.savefig(os.path.join(PLOTS_DIR, 'view_positions_by_gender.png'))
                plt.close()
                logger.info("a tabela e o gráfico mostram a distribuição combinada de gênero e posição. isso pode indicar se um gênero é mais propenso a ter um certo tipo de radiografia.")
            else:
                logger.warning(f"colunas '{PATIENT_GENDER_COL}' ou '{VIEW_POSITION_COL}' não encontradas para análise de crosstab.")

            # 4. relação entre achados comuns e idade/gênero
            logger.info("\n4. relação entre achados comuns e idade/gênero:")
            if all(col in self.original_df.columns for col in [FINDING_LABELS_COL, PATIENT_AGE_COL, PATIENT_GENDER_COL]):
                self.original_df['finding labels list'] = self.original_df[FINDING_LABELS_COL].str.split('|')
                exploded_df = self.original_df.explode('finding labels list')
                exploded_df = exploded_df[exploded_df['finding labels list'] != 'no finding']
                exploded_df[PATIENT_AGE_COL] = pd.to_numeric(exploded_df[PATIENT_AGE_COL], errors='coerce')
                
                average_age_per_finding = exploded_df.groupby('finding labels list')[PATIENT_AGE_COL].mean().sort_values(ascending=False)
                logger.info(f"\nidade média por achado clínico (para achados comuns):\n{average_age_per_finding}")
                
                all_labels_check = self.original_df[FINDING_LABELS_COL].str.split('|').explode()
                label_counts_check = all_labels_check.value_counts()
                label_counts_excluding_nofinding_check = label_counts_check.drop('no finding', errors='ignore')
                common_findings = label_counts_excluding_nofinding_check.head(10).index

                average_age_per_common_finding = average_age_per_finding.loc[average_age_per_finding.index.intersection(common_findings)]

                if not average_age_per_common_finding.empty:
                    plt.figure(figsize=(12, 7))
                    sns.barplot(x=average_age_per_common_finding.index, y=average_age_per_common_finding.values, palette='viridis', hue=average_age_per_common_finding.index, legend=False)
                    plt.title('Idade Média por Achado Clínico (Top 10 Achados)', fontsize=16)
                    plt.xlabel('Achado Clínico', fontsize=12)
                    plt.ylabel('Idade Média do Paciente', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(PLOTS_DIR, 'average_age_per_finding.png'))
                    plt.close()
                else:
                    logger.warning("nenhum dado de idade disponível para os achados comuns selecionados ou colunas necessárias estão faltando.")

                logger.info("\ndistribuição de gênero por achado clínico (para achados comuns):")
                exploded_common_findings_df = exploded_df[exploded_df['finding labels list'].isin(common_findings)].copy()
                if not exploded_common_findings_df.empty:
                    gender_distribution_per_finding = pd.crosstab(exploded_common_findings_df['finding labels list'], exploded_common_findings_df[PATIENT_GENDER_COL], normalize='index')
                    logger.info(f"distribuição de gênero por achado:\n{gender_distribution_per_finding}")
                    gender_distribution_per_finding.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='coolwarm')
                    plt.title('distribuição proporcional de gênero por achado clínico (top 10 achados)', fontsize=16)
                    plt.xlabel('achado clínico', fontsize=12)
                    plt.ylabel('proporção', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.legend(title='gênero')
                    plt.tight_layout()
                    plt.savefig(os.path.join(PLOTS_DIR, 'gender_distribution_per_finding.png'))
                    plt.close()
                else:
                    logger.warning("nenhum dado de gênero disponível para os achados comuns selecionados.")
            else:
                logger.warning(f"colunas '{FINDING_LABELS_COL}', '{PATIENT_AGE_COL}' ou '{PATIENT_GENDER_COL}' não encontradas para análise de achados vs idade/gênero.")
        else:
            logger.warning("dataframe não carregado. pulando análise de correlação.")

    def run_analysis(self):
        """orquestra todo o pipeline de análise descritiva."""
        os.makedirs(PLOTS_DIR, exist_ok=True)
        self.load_data()
        if self.original_df is not None:
            self.display_general_info()
            self.analyze_finding_labels()
            self.analyze_categorical_columns()
            self.analyze_patient_age()
            self.count_unique_identifiers()
            self.analyze_correlations()
        else:
            logger.error("análise descritiva abortada devido a falha no carregamento dos dados originais.")

if __name__ == "__main__":
    logger.info("executando exemplo de descriptive statistics...")
    stats_analyzer = DescriptiveStatistics()
    stats_analyzer.run_analysis()