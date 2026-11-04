from __future__ import annotations

import re
import unittest
from pathlib import Path


FRONTEND_FOREST_DIR = Path(__file__).resolve().parents[1] / "frontend" / "forest"
STATE_IMPORT_RE = re.compile(r'from\s+["\'](\./state\.js\?v=[^"\']+)["\']')


class FrontendModuleGraphTest(unittest.TestCase):
    def test_forest_modules_share_one_state_module_url(self) -> None:
        imports: list[tuple[str, str]] = []
        for module_path in sorted(FRONTEND_FOREST_DIR.glob("*.js")):
            source = module_path.read_text(encoding="utf-8")
            imports.extend(
                (module_path.name, match.group(1))
                for match in STATE_IMPORT_RE.finditer(source)
            )

        self.assertGreaterEqual(
            len(imports),
            2,
            "Expected multiple forest modules to import shared state",
        )
        import_urls = {import_url for _, import_url in imports}
        self.assertEqual(
            len(import_urls),
            1,
            "Forest modules must use the exact same state.js URL so the browser "
            f"keeps one forestState singleton; found: {imports}",
        )


if __name__ == "__main__":
    unittest.main()
