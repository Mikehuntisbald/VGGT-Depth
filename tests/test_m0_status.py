from pathlib import Path

from tools._m0_status import Status, atomic_write_json, sha256_file


def test_atomic_receipt_and_hash(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "receipt.json"
    atomic_write_json(output, {"status": Status.PASS})
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert len(sha256_file(output)) == 64


def test_status_values_are_stable() -> None:
    assert [status.value for status in Status] == [
        "PASS",
        "PASS_WITH_FALLBACK",
        "FAIL",
        "BLOCKED",
        "NOT_RUN",
    ]

