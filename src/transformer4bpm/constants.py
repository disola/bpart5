from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).parents[2].resolve()

DATA_ROOT = PROJECT_ROOT / "data"
DATA_PATH_TO_CSV_FILES = DATA_ROOT / "raw/sapsam_2022/csv"
DATA_PATH_TO_XML_FILES = DATA_ROOT / "raw/sapsam_2022/xml"

DATA_INTERIM = DATA_ROOT / "interim" 
BPMN_MODEL_META_DATA = DATA_INTERIM / "bpmn_model_meta_data.pkl"
DATA_CACHE = DATA_INTERIM / "cache"

MODELS = PROJECT_ROOT / "models"

REPORTS = PROJECT_ROOT / "reports"
RECOMMENDATION_ROOT = REPORTS / "recommendation/bpart5"

TRAIN_SEQUENCES = DATA_INTERIM / "train.json"
VALIDATE_SEQUENCES = DATA_INTERIM / "validate.json"
TEST_SEQUENCES = DATA_INTERIM / "test.json"

PROCESSES_SPLIT_FILE = DATA_INTERIM / "processes_split.json"
SIMULATED_MODELS_FILE = DATA_INTERIM / "simulated_models.json"
RECOMMENDATION_CASES_FILE = DATA_INTERIM / "recommendation_cases.json"

SRC_ROOT = PROJECT_ROOT / "src"
EXAMPLE_PROCESSES = SRC_ROOT / "prefilled_example_processes.json"

BPMN2_NAMESPACE = "http://b3mn.org/stencilset/bpmn2.0#"

XML_FILE_EXTENSION = ".xml"