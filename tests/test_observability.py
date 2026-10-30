"""Structured event-stream tests."""

import json

from control.observability import append_event, event, new_run_id


def test_events_are_structured_and_append_only(tmp_path):
    run_id = new_run_id()
    first = event(run_id, "session", duration_s=0.25, tools=2)
    path = append_event(first, tmp_path)
    assert path == tmp_path / "report" / "events.jsonl"

    second = event(run_id, "referee", decision="benign", counted=1)
    append_event(second, tmp_path)

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert all(record["run_id"] == run_id for record in records)
    assert all(record["schema_version"] == 1 for record in records)
    assert all({"event_id", "run_id", "ts", "stage", "status"}
               <= set(record) for record in records)
    assert records[0]["stage"] == "session"
    assert records[0]["tools"] == 2
    assert records[1]["decision"] == "benign"
    assert "steer" not in records[0]
