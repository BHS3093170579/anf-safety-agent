"""多路页「开始监控」实跑一次：验证 fragment 更新画面不报错。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> int:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=900)
    at.run()
    if at.exception:
        print("初始异常:", [e.message[:200] for e in at.exception])
        return 1

    print("按钮数:", len(at.button))
    starts = [b for b in at.button if "开始监控" in (b.label or "")]
    if not starts:
        print("找不到开始按钮")
        return 1
    at = starts[0].click().run()

    errs = [e.message for e in at.exception]
    print("点击后异常数:", len(errs))
    for e in errs[:3]:
        print("  !", e[:300])

    st_ = at.session_state.get("multi") or {}
    print("running:", st_.get("running"), "已跑帧:", st_.get("i"))
    print("告警:", st_.get("alerts"))
    return 1 if errs else 0

if __name__ == "__main__":
    raise SystemExit(main())
