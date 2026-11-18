"""一键冒烟：依次跑 tests 下的回归脚本，输出汇总。

用法：
    python tests/run_all.py            # 跑全部（跳过 bench 与需要摄像头的用例）
    python tests/run_all.py --with-cam # 额外跑需要现场摄像头的用例
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SKIP_DEFAULT = {"bench.py", "run_all.py", "smoke_multi_rtsp.py"}
SKIP_CAM = {"smoke_multi_rtsp.py", "smoke_rtsp.py"}


def main() -> int:
    with_cam = "--with-cam" in sys.argv
    skip = SKIP_DEFAULT if with_cam else (SKIP_DEFAULT | SKIP_CAM)

    cases = sorted(p for p in HERE.glob("*.py") if p.name not in skip)
    print(f"共 {len(cases)} 个用例\n")

    rows = []
    t0 = time.time()
    for p in cases:
        start = time.time()
        proc = subprocess.run(
            [sys.executable, str(p)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        cost = time.time() - start
        ok = proc.returncode == 0
        rows.append((p.name, cost, proc.returncode, ok))
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {p.name:<24} {cost:6.1f}s  exit={proc.returncode}")
        if not ok:
            tail = (proc.stdout or "")[-800:]
            err = (proc.stderr or "")[-800:]
            print(tail)
            print(err)

    passed = sum(1 for r in rows if r[3])
    print(f"\n通过 {passed}/{len(rows)}，总耗时 {time.time() - t0:.1f}s")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
