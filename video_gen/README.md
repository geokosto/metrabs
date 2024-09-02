### To generate poses without visualization:
```bash
python -m video_gen.metrabs_get_video path/to/your/video.mp4 1 20
```

### To generate poses with visualization:
```bash
python -m video_gen.metrabs_get_video path/to/your/video.mp4 1 20 --visualize
```

### To load pre-existing poses without visualization:
```bash
python -m video_gen.metrabs_get_video path/to/your/video.mp4 1 20 --load_poses
```

### To load pre-existing poses with visualization:
```bash
python -m video_gen.metrabs_get_video path/to/your/video.mp4 1 20 --load_poses --visualize

python -m video_gen.metrabs_get_video nan_dunk_crop.mp4 8 --load_poses --visualize
```


python -m video_gen.metrabs_get_video lessort-dunk-01.mp4 2 --visualize
