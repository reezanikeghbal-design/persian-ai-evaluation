from app.core.text import normalize_persian
from app.evaluation.dataset import validate_dataset, load_jsonl


def test_persian_normalization():
    assert normalize_persian("كتاب  و  مي‌روم") == "کتاب و می‌روم"


def test_dataset_validation():
    valid = {"id": "1", "category": "x", "question": "سؤال", "reference_answer": "پاسخ",
             "context": "زمینه", "difficulty": "easy"}
    assert validate_dataset([valid]) == []
    assert any("duplicate" in error for error in validate_dataset([valid, valid]))
    invalid = dict(valid, question="", reference_answer="")
    assert any("empty question" in error for error in validate_dataset([invalid]))


def test_golden_dataset_has_50_plus_records():
    records = load_jsonl("datasets/golden_dataset.jsonl")
    assert len(records) >= 50
    assert validate_dataset(records) == []


def test_invalid_record_structure_is_reported():
    assert any("must be an object" in error for error in validate_dataset(["not-an-object"]))
    assert any("missing fields" in error for error in validate_dataset([{"id": "incomplete"}]))
