# 发布到 GitHub

仓库已经初始化好并完成首次提交，下面只差推送到远端和发布权重包。

## 1. 新建远端仓库

在 GitHub 上新建一个空仓库（**不要**勾选 README、.gitignore、License，本地已有），仓库名建议 `anf-safety-agent`。

## 2. 关联并推送

```bash
cd <仓库目录>

git remote add origin https://github.com/<你的用户名>/anf-safety-agent.git
git branch -M main
git push -u origin main
```

推送后确认：仓库首页能看到 README 的双栏截图（`docs/images/helmet.png`、`fire.png`）和 mermaid 架构图。mermaid 需要 GitHub 原生渲染，在网页上会自动画出来。

## 3. 发布权重（Releases）

权重和演示视频不进 Git 仓库（`.gitignore` 已排除），走 Release 附件。GitHub 单个附件上限 2 GB，当前四个权重包合计约 200 MB，可以直接上传。

需要打包的文件：

```
weights/
├── helmet/     best.engine  best.pt  best.onnx
├── fire/       best.engine  best.pt  best.onnx
├── pose/       yolov8n-pose.pt
└── helmet_qc/  best.pt  head_det.pt  head_cls.pt  yolov8n-pose.pt
```

本地打 zip：

```powershell
Compress-Archive -Path weights\* -DestinationPath weights-v1.0.zip
```

然后：

1. 仓库右侧 **Releases → Create a new release**
2. Tag 填 `v1.0.0`，标题填 `v1.0.0 — 五合一工地安全检测`
3. 把 `weights-v1.0.zip` 拖进附件区
4. 发布

README 里的 `[Releases](../../releases)` 链接会直接指向这个页面。

## 4. 仓库设置建议

| 项目 | 建议值 |
|---|---|
| Description | 工地安全实时视频分析：安全帽 / 火焰烟雾 / 睡岗 / 离岗 / 徘徊，五合一边缘推理 |
| Topics | `computer-vision` `yolo` `tensorrt` `streamlit` `construction-safety` `object-detection` `pose-estimation` `bytetrack` |
| License | MIT（已随仓库提供 `LICENSE`） |
| 默认分支 | `main` |

## 5. 推送前自检

```bash
git status                      # 工作区应为 clean
git ls-files | wc -l            # 约 70 个文件
du -sh .git                     # 约 2 MB，不应出现大文件
```

如果误把权重加进了暂存区，用：

```bash
git rm --cached -r weights/*.engine weights/*.onnx weights/*.pt
```
