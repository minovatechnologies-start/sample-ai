from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from minova_ai.api import app


class APITests(unittest.TestCase):
    def test_service_starts_degraded_when_adapter_is_missing(self) -> None:
        with (
            patch.dict(os.environ, {"API_KEY": "test-api-key", "ADAPTER_PATH": ""}, clear=True),
            patch("minova_ai.api._load_model") as load_model,
            TestClient(app) as client,
        ):
            health = client.get("/health")
            generation = client.post(
                "/generate",
                json={"prompt": "Write a function.", "language": "python"},
                headers={"Authorization": "Bearer test-api-key"},
            )

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "degraded")
        self.assertEqual(health.json()["model"], "not_loaded")
        self.assertIn("ADAPTER_PATH", health.json()["detail"])
        self.assertEqual(generation.status_code, 503)
        self.assertIn("ADAPTER_PATH", generation.json()["detail"])
        load_model.assert_not_called()

    def test_service_loads_model_when_adapter_directory_exists(self) -> None:
        with tempfile.TemporaryDirectory() as adapter_directory:
            model = object()
            tokenizer = object()
            with (
                patch.dict(
                    os.environ,
                    {"API_KEY": "test-api-key", "ADAPTER_PATH": str(Path(adapter_directory))},
                    clear=True,
                ),
                patch("minova_ai.api._load_model", return_value=(model, tokenizer, "reviewed/model")) as load_model,
                TestClient(app) as client,
            ):
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model": "reviewed/model"})
        load_model.assert_called_once_with()
