### To generate poses without visualization:
```bash
python -m video_gen.metrabs_get_video case_name video_name 1 20
```

### To generate poses with visualization:
```bash
python -m video_gen.metrabs_get_video case_name video_name 1 20 --visualize
```

### To load pre-existing poses without visualization:
```bash
python -m video_gen.metrabs_get_video case_name video_name 1 20 --load_poses
```

### To load pre-existing poses with visualization:
```bash
python -m video_gen.metrabs_get_video case_name video_name 1 20 --load_poses --visualize

python -m video_gen.metrabs_get_video pao_promo nan_dunk_crop.mp4 8 --load_poses --visualize
```

## Examples
```bash
python -m video_gen.metrabs_get_video pao_promo lessort-dunk-01.mp4 2 --visualize

python -m video_gen.metrabs_get_image_dev pao_promo sloukas-shot-01.mp4 6 --visualize --camera_data sloukas-shot-01_camera_calibration.npz

```