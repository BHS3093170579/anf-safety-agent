"""多路监控页签冒烟：只验证控件与渲染，不点「开始」（持续模式会一直 rerun）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()

    errs = [e.message for e in at.exception]
    print("异常数:", len(errs))
    for e in errs:
        print("  !", e[:300])

    labels = [b.label for b in at.button]
    print("包含开始监控:", any("开始监控" in x for x in labels))
    print("包含暂停:", any("暂停" in x for x in labels))
    print("能力下拉:", [s.key for s in at.selectbox if "multi_cap" in (s.key or "")])
    print("路数滑块:", [s.key for s in at.slider if s.key == "multi_n"])
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
