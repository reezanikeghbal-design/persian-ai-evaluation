from app.evaluation.metrics import recall_at_k, mean_reciprocal_rank, answer_relevancy, faithfulness_heuristic


def test_recall_at_k():
    assert recall_at_k(["a", "b"], {"a", "c"}, 2) == 0.5


def test_mrr():
    assert mean_reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5


def test_answer_relevancy():
    assert answer_relevancy("هوش مصنوعی چیست؟", "هوش مصنوعی شاخه‌ای از علوم رایانه است.") > 0


def test_faithfulness_heuristic():
    assert faithfulness_heuristic("Docker از کانتینر استفاده می‌کند", "Docker از کانتینر استفاده می‌کند") == 1
    assert faithfulness_heuristic("ادعای کاملاً نامرتبط", "متن دیگری") < 1
