import sys
import unittest
from importlib.util import find_spec
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if find_spec("flask") is not None:
    from app import create_app
else:
    create_app = None


@unittest.skipIf(create_app is None, "当前 Python 环境缺少 Flask")
class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app().test_client()

    def test_health_endpoint(self):
        response = self.app.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")

    def test_tasks_endpoint(self):
        response = self.app.get("/api/tasks")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("scoliosis", data["tasks"])
        self.assertIn("slippage", data["tasks"])

    def test_scoliosis_requires_image(self):
        response = self.app.post("/api/scoliosis/keypoints", data={})
        self.assertEqual(response.status_code, 400)

    def test_slippage_requires_image(self):
        response = self.app.post("/api/slippage/keypoints", data={})
        self.assertEqual(response.status_code, 400)

    def test_scoliosis_analysis_requires_keypoints(self):
        response = self.app.post("/api/scoliosis/analysis", json={})
        self.assertEqual(response.status_code, 400)

    def test_slippage_analysis_requires_keypoints(self):
        response = self.app.post("/api/slippage/analysis", json={})
        self.assertEqual(response.status_code, 400)

    def test_preview_requires_file(self):
        response = self.app.post("/api/preview", data={})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
