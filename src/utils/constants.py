import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')


DATA_ENTRY_FILENAME = 'Data_Entry_2017.csv'
SOURCE_DICTIONARY_FILENAME = 'source_dictionary.csv'
PROCESSED_DATA_FILENAME = 'data_entry.csv'
DATA_DICTIONARY_FILENAME = 'data_dictionary.csv'

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
TARGET_WIDTH = 256
TARGET_HEIGHT = 256
VIEW_POSITION_DIR_MAP = {0: 'PA', 1: 'AP'}
FINDING_LABELS_DIR_MAP = {0: '0', 1: '1'}

IMAGE_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
VIEW_POSITION_FOR_MODEL = 'PA'
TARGET_LABELS = ['0', '1']
MODEL_CHECKPOINT_FILENAME = 'best_model_pa_effusion.keras'

IMAGE_INDEX_COL = 'Image Index'
FINDING_LABELS_COL = 'Finding Labels'
PATIENT_ID_COL = 'Patient ID'
PATIENT_AGE_COL = 'Patient Age'
PATIENT_GENDER_COL = 'Patient Gender'
VIEW_POSITION_COL = 'View Position'
ORIGINAL_IMAGE_WIDTH_COL = 'OriginalImageWidth'
ORIGINAL_IMAGE_HEIGHT_COL = 'OriginalImageHeight'
ORIGINAL_IMAGE_PIXEL_SPACING_X_COL = 'OriginalImagePixelSpacing_x'
ORIGINAL_IMAGE_PIXEL_SPACING_Y_COL = 'OriginalImagePixelSpacing_y'
UNNAMED_COL = 'Unnamed: 11'
STRATIFY_COL = 'stratify_col'
SOURCE_PATH_COL = 'source_path'

EVALUATION_CONFUSION_MATRIX_FILENAME = 'evaluation_confusion_matrix.png'
EVALUATION_CLASSIFICATION_REPORT_FILENAME = 'evaluation_classification_report.txt'