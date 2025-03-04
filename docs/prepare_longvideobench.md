****## Prepare LongVideoBench Dataset


### Step 1: Download LongVideoBench dataset from [huggingface](git clone https://huggingface.co/datasets/longvideobench/LongVideoBench)
```bash
git clone git clone https://huggingface.co/datasets/longvideobench/LongVideoBench
```

Denote the root directory of download LongVideoBench dataset as `longvideobench_root`, it should has the following structure:
```
${longvideobench_root}/
├── subtitles.zip
├── test-00000-of-00001.parquet
├── validation-00000-of-00001.parquet
├── videos.tar.part.aa
├── ...
└── videos.tar.part.be
├── ...
```


### Step 2: Unzip everything
```bash
cd ${longvideobench_root}
unzip subtitle.zip
cat videos.tar.part.* > videos.tar
tar -xzvf videos.tar
```


### Step 3: Extract frames of all videos
```bash
cd ${repo_root}
python scripts/utils/frame_extraction.py \
--videofile_tpl ${longvideobench_root}/data/'*.mp4' \
--results_dir ${longvideobench_root}/video_25fps \
--fps 25 \
--num_workers 32
```


### Step 4: Build LongVideoBench dataset
```bash
cd ${repo_root}
python scripts/utils/build_longvideobench_dataset.py \
--hf_root ${longvideobench_root} \
--hf_qwen2vl7b_path ${PATH_TO_Qwen2-VL-7B-Instruct}
```
Note that you can NOT modify folder `${longvideobench_root}/video_25fps` after this step, since the absolute path of extracted frames are written into annotation files `longvideobench_val.json` and `longvideobench_test.json`:
```
repo_root/
├── dataset/
    ├── longvideobench/
        ├── longvideobench_val.json
        ├── longvideobench_test.json
├── ...
```