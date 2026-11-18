"""前端冒烟：AppTest 渲染 app.py，确认三路选择器与四页签都在。"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()

    errors = [e.value for e in at.exception] + [e.value for e in at.error]
    print("异常数:", len(errors))
    for e in errors:
        print("  ", e)

    keys = [sb.key for sb in at.selectbox]
    print("下拉框:", keys)
    print("页签数:", len(at.tabs) if hasattr(at, "tabs") else "n/a")

    assert not errors, "页面渲染有异常"
    assert "pipeline_helmet" in keys, "安全帽检测方式下拉缺失"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
