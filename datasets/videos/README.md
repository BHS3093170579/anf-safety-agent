# 演示视频

视频不进仓库（体积大、可能含现场隐私），准备任意 mp4 放到对应子目录即可。

```
datasets/videos/
├── helmet/   # 工地安全帽场景，建议 1280x720 以上
├── fire/     # 火焰 / 烟雾场景
├── sleep/    # 值班室、趴桌睡觉
└── loiter/   # 大门、通道，有人来回走动
```

`config/cameras.yaml` 的 `fallback` 段指定了每个相机在缺少 RTSP 时使用的本地文件，把文件名对上就能直接跑通四路演示：

```yaml
fallback:
  Camera_01: "datasets/videos/helmet/demo.mp4"
  Camera_02: "datasets/videos/fire/demo.mp4"
  Camera_03: "datasets/videos/sleep/demo.mp4"
  Camera_04: "datasets/videos/loiter/demo.mp4"
```

也可以只放一个视频，用「实时监控」页单独跑某个场景。
