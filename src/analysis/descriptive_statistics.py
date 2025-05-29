import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils.logger import logger
from src.utils.constants import DATA_ENTRY_FILENAME, FINDING_LABELS_COL, VIEW_POSITION_COL, PATIENT_GENDER_COL, PATIENT_AGE_COL, IMAGE_INDEX_COL, PATIENT_ID_COL

PLOTS_DIR = 'reports'

sns.set_style('whitegrid')

class DescriptiveStatistics:
    """
    Realiza análise estatística descritiva e gera visualizações
    para o conjunto de dados original de raios-X de tórax do NIH.
    """
    def __init__(self, data_entry_filename=DATA_ENTRY_FILENAME):
        from src.data.data_loader import DataLoader
        self.data_loader = DataLoader(data_entry_filename=data_entry_filename)
        self.original_df = None
        logger.info("DescriptiveStatistics initialized.")

    def load_data(self):
        """
        Carrega o arquivo CSV de entrada de dados original.
        """
        logger.info("Attempting to load original data for descriptive analysis using DataLoader...")
        self.original_df = self.data_loader.load_original_data()
        if self.original_df is not None:
            logger.info(f"Dataset loaded successfully! Total entries: {len(self.original_df)}")
            logger.info("\nFirst 5 rows of the dataset:\n" + str(self.original_df.head()))
        else:
            logger.error("Failed to load original data for descriptive analysis.")

    def display_general_info(self):
        """
        Exibe informações gerais e estatísticas descritivas do DataFrame.
        """
        if self.original_df is not None:
            logger.info("\nGeneral information of the original DataFrame:")
            self.original_df.info()
            logger.info("\nDescriptive statistics for numerical columns:")
            numerical_cols = [
                'Follow-up #', PATIENT_AGE_COL, 'OriginalImage[Width',
                'Height]', 'OriginalImagePixelSpacing[x', 'y]'
            ]
            existing_numerical_cols = [col for col in numerical_cols if col in self.original_df.columns]
            if existing_numerical_cols:
                logger.info(self.original_df[existing_numerical_cols].describe())
            else:
                logger.warning("No suitable numerical columns found for descriptive statistics.")
        else:
            logger.warning("DataFrame not loaded. Skipping general info display.")

    def analyze_finding_labels(self):
        """
        Analisa e visualiza a distribuição de 'Finding Labels'.
        """
        if self.original_df is not None and FINDING_LABELS_COL in self.original_df.columns:
            logger.info(f"\nAnalyzing column '{FINDING_LABELS_COL}':")
            all_labels = self.original_df[FINDING_LABELS_COL].str.split('|').explode()
            label_counts = all_labels.value_counts()
            logger.info(f"Count of occurrences for each finding or condition:\n{label_counts}")

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
                logger.warning("No findings other than 'No Finding' to plot or 'No Finding' is the only label.")
        else:
            logger.warning(f"DataFrame not loaded or column '{FINDING_LABELS_COL}' not found. Skipping 'Finding Labels' analysis.")

    def analyze_categorical_columns(self):
        """
        Analisa e visualiza 'View Position' e 'Patient Gender'.
        """
        if self.original_df is not None:
            if VIEW_POSITION_COL in self.original_df.columns:
                logger.info(f"\nAnalyzing column '{VIEW_POSITION_COL}':")
                position_counts = self.original_df[VIEW_POSITION_COL].value_counts()
                logger.info(f"Count of occurrences for each view position:\n{position_counts}")
                plt.figure(figsize=(6, 5))
                sns.countplot(data=self.original_df, x=VIEW_POSITION_COL, hue=VIEW_POSITION_COL, palette='pastel', legend=False)
                plt.title('Distribuição da Posição de Visualização', fontsize=14)
                plt.xlabel('Posição de Visualização', fontsize=12)
                plt.ylabel('Número de Imagens', fontsize=12)
                plt.savefig(os.path.join(PLOTS_DIR, 'view_position_distribution.png'))
                plt.close()
            else:
                logger.warning(f"Column '{VIEW_POSITION_COL}' not found. Skipping its analysis.")

            if PATIENT_GENDER_COL in self.original_df.columns:
                logger.info(f"\nAnalyzing column '{PATIENT_GENDER_COL}':")
                gender_counts = self.original_df[PATIENT_GENDER_COL].value_counts()
                logger.info(f"Count of occurrences for each gender:\n{gender_counts}")
                plt.figure(figsize=(6, 5))
                sns.countplot(data=self.original_df, x=PATIENT_GENDER_COL, hue=PATIENT_GENDER_COL, palette='coolwarm', legend=False)
                plt.title('Distribuição de Gênero do Paciente', fontsize=14)
                plt.xlabel('Gênero', fontsize=12)
                plt.ylabel('Número de Imagens', fontsize=12)
                plt.savefig(os.path.join(PLOTS_DIR, 'patient_gender_distribution.png'))
                plt.close()
            else:
                logger.warning(f"Column '{PATIENT_GENDER_COL}' not found. Skipping its analysis.")
        else:
            logger.warning("DataFrame not loaded. Skipping categorical columns analysis.")

    def analyze_patient_age(self):
        """
        Analisa e visualiza a distribuição de 'Patient Age'.
        """
        if self.original_df is not None and PATIENT_AGE_COL in self.original_df.columns:
            logger.info(f"\nAnalyzing column '{PATIENT_AGE_COL}':")
            self.original_df[PATIENT_AGE_COL] = pd.to_numeric(self.original_df[PATIENT_AGE_COL], errors='coerce')
            logger.info(f"Descriptive statistics for Patient Age:\n{self.original_df[PATIENT_AGE_COL].describe()}")

            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.original_df, x=PATIENT_AGE_COL, bins=30, kde=True, color='skyblue')
            
            # Ajusta os limites do eixo x dinamicamente ou define um máximo razoável
            min_age = self.original_df[PATIENT_AGE_COL].min()
            max_age = self.original_df[PATIENT_AGE_COL].max()
            if min_age is not None and max_age is not None:
                plt.xlim(min_age - 5 if min_age > 5 else 0, max_age + 5 if max_age < 120 else 120)
            else:
                plt.xlim(0, 100) # Default if min/max are None

            plt.title('Distribuição da Idade do Paciente', fontsize=16)
            plt.xlabel('Idade', fontsize=12)
            plt.ylabel('Frequência', fontsize=12)
            plt.savefig(os.path.join(PLOTS_DIR, 'patient_age_distribution.png'))
            plt.close()
        else:
            logger.warning(f"DataFrame not loaded or column '{PATIENT_AGE_COL}' not found. Skipping 'Patient Age' analysis.")

    def count_unique_identifiers(self):
        """
        Conta e registra o número de imagens e pacientes únicos.
        """
        if self.original_df is not None:
            num_total_entries = len(self.original_df)
            num_unique_images = self.original_df[IMAGE_INDEX_COL].nunique() if IMAGE_INDEX_COL in self.original_df.columns else 0
            num_unique_patients = self.original_df[PATIENT_ID_COL].nunique() if PATIENT_ID_COL in self.original_df.columns else 0

            logger.info("\nCounting Unique Images and Patients:")
            logger.info(f"Total entries in CSV: {num_total_entries}")
            logger.info(f"Number of unique image file names: {num_unique_images}")
            logger.info(f"Number of unique patient IDs: {num_unique_patients}")
        else:
            logger.warning("DataFrame not loaded. Skipping unique identifiers count.")

    def analyze_correlations(self):
        """
        Analisa e visualiza correlações entre atributos.
        """
        if self.original_df is not None:
            logger.info("\n--- Correlation Analysis ---")

            # 1. Correlação entre variáveis numéricas
            logger.info("\n1. Correlação entre variáveis numéricas (Pearson):")
            numerical_cols_to_check = [
                'Follow-up #', PATIENT_AGE_COL, 'OriginalImage[Width',
                'Height]', 'OriginalImagePixelSpacing[x', 'y]'
            ]
            numerical_cols = [col for col in numerical_cols_to_check if col in self.original_df.columns]

            if PATIENT_AGE_COL in self.original_df.columns:
                self.original_df[PATIENT_AGE_COL] = pd.to_numeric(self.original_df[PATIENT_AGE_COL], errors='coerce')

            if numerical_cols:
                correlation_matrix = self.original_df[numerical_cols].dropna().corr()
                logger.info(f"Correlation Matrix:\n{correlation_matrix}")
                plt.figure(figsize=(len(numerical_cols)*2, len(numerical_cols)*2))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
                plt.title('Matriz de Correlação (Numérica)', fontsize=14)
                plt.savefig(os.path.join(PLOTS_DIR, 'correlation_matrix_numerical.png'))
                plt.close()
            else:
                logger.warning("No suitable numerical columns to calculate correlation matrix.")

            # 2. Relação entre Idade do Paciente e Gênero/Posição de Visualização
            logger.info("\n2. Relação entre Idade do Paciente e Gênero/Posição de Visualização:")
            if PATIENT_GENDER_COL in self.original_df.columns and PATIENT_AGE_COL in self.original_df.columns:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=self.original_df, x=PATIENT_GENDER_COL, y=PATIENT_AGE_COL, palette='coolwarm', hue=PATIENT_GENDER_COL, legend=False)
                plt.title('Distribuição de Idade por Gênero', fontsize=14)
                plt.xlabel('Gênero', fontsize=12)
                plt.ylabel('Idade do Paciente', fontsize=12)
                plt.savefig(os.path.join(PLOTS_DIR, 'age_distribution_by_gender.png'))
                plt.close()
            else:
                logger.warning(f"Columns '{PATIENT_GENDER_COL}' or '{PATIENT_AGE_COL}' not found for plotting Age by Gender.")
            
            if VIEW_POSITION_COL in self.original_df.columns and PATIENT_AGE_COL in self.original_df.columns:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=self.original_df, x=VIEW_POSITION_COL, y=PATIENT_AGE_COL, palette='pastel', hue=VIEW_POSITION_COL, legend=False)
                plt.title('Distribuição de Idade por Posição de Visualização', fontsize=14)
                plt.xlabel('Posição de Visualização', fontsize=12)
                plt.ylabel('Idade do Paciente', fontsize=12)
                plt.savefig(os.path.join(PLOTS_DIR, 'age_distribution_by_view_position.png'))
                plt.close()
            else:
                logger.warning(f"Columns '{VIEW_POSITION_COL}' or '{PATIENT_AGE_COL}' not found for plotting Age by View Position.")
            logger.info("Nota: Boxplots mostram mediana, quartis e potenciais outliers na idade para cada categoria.")

            # 3. Relação entre Gênero e Posição de Visualização
            logger.info("\n3. Relação entre Gênero e Posição de Visualização:")
            if PATIENT_GENDER_COL in self.original_df.columns and VIEW_POSITION_COL in self.original_df.columns:
                gender_position_crosstab = pd.crosstab(self.original_df[PATIENT_GENDER_COL], self.original_df[VIEW_POSITION_COL])
                logger.info(f"Crosstab (Gender vs View Position):\n{gender_position_crosstab}")
                plt.figure(figsize=(8, 6))
                sns.countplot(data=self.original_df, x=VIEW_POSITION_COL, hue=PATIENT_GENDER_COL, palette='viridis')
                plt.title('Contagem de Posições de Visualização por Gênero', fontsize=14)
                plt.xlabel('Posição de Visualização', fontsize=12)
                plt.ylabel('Número de Imagens', fontsize=12)
                plt.legend(title='Gênero')
                plt.savefig(os.path.join(PLOTS_DIR, 'view_positions_by_gender.png'))
                plt.close()
                logger.info("Nota: A tabela e o gráfico mostram a distribuição combinada de gênero e posição. Isso pode indicar se um gênero é mais propenso a ter um certo tipo de radiografia.")
            else:
                logger.warning(f"Columns '{PATIENT_GENDER_COL}' or '{VIEW_POSITION_COL}' not found for crosstab analysis.")

            # 4. Relação entre Achados Comuns e Idade/Gênero
            logger.info("\n4. Relação entre Achados Comuns e Idade/Gênero:")
            if all(col in self.original_df.columns for col in [FINDING_LABELS_COL, PATIENT_AGE_COL, PATIENT_GENDER_COL]):
                self.original_df['Finding Labels List'] = self.original_df[FINDING_LABELS_COL].str.split('|')
                exploded_df = self.original_df.explode('Finding Labels List')
                exploded_df = exploded_df[exploded_df['Finding Labels List'] != 'No Finding']
                exploded_df[PATIENT_AGE_COL] = pd.to_numeric(exploded_df[PATIENT_AGE_COL], errors='coerce')
                
                average_age_per_finding = exploded_df.groupby('Finding Labels List')[PATIENT_AGE_COL].mean().sort_values(ascending=False)
                logger.info(f"\nAverage Age per Clinical Finding (for common findings):\n{average_age_per_finding}")
                
                all_labels_check = self.original_df[FINDING_LABELS_COL].str.split('|').explode()
                label_counts_check = all_labels_check.value_counts()
                label_counts_excluding_nofinding_check = label_counts_check.drop('No Finding', errors='ignore')
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
                    logger.warning("No age data available for selected common findings or required columns are missing.")

                logger.info("\nGender Distribution per Clinical Finding (for common findings):")
                exploded_common_findings_df = exploded_df[exploded_df['Finding Labels List'].isin(common_findings)].copy()
                if not exploded_common_findings_df.empty:
                    gender_distribution_per_finding = pd.crosstab(exploded_common_findings_df['Finding Labels List'], exploded_common_findings_df[PATIENT_GENDER_COL], normalize='index')
                    logger.info(f"Gender distribution per finding:\n{gender_distribution_per_finding}")
                    gender_distribution_per_finding.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='coolwarm')
                    plt.title('Distribuição Proporcional de Gênero por Achado Clínico (Top 10 Achados)', fontsize=16)
                    plt.xlabel('Achado Clínico', fontsize=12)
                    plt.ylabel('Proporção', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.legend(title='Gênero')
                    plt.tight_layout()
                    plt.savefig(os.path.join(PLOTS_DIR, 'gender_distribution_per_finding.png'))
                    plt.close()
                else:
                    logger.warning("No gender data available for selected common findings.")
            else:
                logger.warning(f"Columns '{FINDING_LABELS_COL}', '{PATIENT_AGE_COL}' or '{PATIENT_GENDER_COL}' not found for findings vs age/gender analysis.")
        else:
            logger.warning("DataFrame not loaded. Skipping correlation analysis.")

    def run_analysis(self):
        """
        Orquestra todo o pipeline de análise descritiva.
        """
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
            logger.error("Descriptive analysis aborted due to failure in loading original data.")

if __name__ == "__main__":
    logger.info("Running DescriptiveStatistics example...")
    stats_analyzer = DescriptiveStatistics()
    stats_analyzer.run_analysis()