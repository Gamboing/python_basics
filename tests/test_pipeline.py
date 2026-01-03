import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from run import build_config, parse_args
from src.config import ExportConfig, mode_defaults


class DummyDetector:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image):
        return []  # sin detecciones


@unittest.skipUnless(cv2 and np, "OpenCV y NumPy requeridos para pruebas de video")
class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)
        self.video_path = self.tmp_path / "test.mp4"
        self.model_path = self.tmp_path / "model.onnx"
        self.model_path.touch()
        self._make_video(self.video_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _make_video(self, path: Path) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(3):
            writer.write(frame)
        writer.release()

    def test_cli_parses_arguments(self):
        args = parse_args(
            [
                "--model-path",
                str(self.model_path),
                "--source",
                str(self.video_path),
                "--mode",
                "fast",
                "--dry-run",
            ]
        )
        self.assertEqual(args.model_path, self.model_path)
        self.assertEqual(args.source, str(self.video_path))
        self.assertTrue(args.dry_run)

    def test_pipeline_runs_and_exports_empty_files(self):
        json_out = self.tmp_path / "out.json"
        csv_out = self.tmp_path / "out.csv"
        defaults = mode_defaults("fast")
        config = replace(
            defaults,
            model_path=self.model_path,
            source=str(self.video_path),
            output=None,
            max_frames=5,
            dry_run=True,
            export=ExportConfig(json_path=json_out, csv_path=csv_out),
        )
        # parchea el detector para no depender de ONNXRuntime
        from src import pipeline as pipeline_module

        pipeline_module.OnnxDetector = DummyDetector
        from src.pipeline import Pipeline

        pipeline = Pipeline(config)
        pipeline.run()

        # El pipeline debe haber creado archivos aunque no haya detecciones
        self.assertTrue(json_out.exists())
        with json_out.open() as f:
            data = json.load(f)
        self.assertEqual(data, [])

        self.assertTrue(csv_out.exists())
        self.assertEqual(os.path.getsize(csv_out), 0)


if __name__ == "__main__":
    unittest.main()
