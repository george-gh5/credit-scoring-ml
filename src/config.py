import pathlib

# корень проекта
BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()

# путь к исходным данным
PATH = BASE_DIR / 'data' / 'raw' / 'cs-training.csv'

# папка для обработанных данных
PROCESSED_DATA_PATH = BASE_DIR / 'data' / 'processed'

# папка для сохранения моделей
MODELS_PATH = BASE_DIR / 'models'
