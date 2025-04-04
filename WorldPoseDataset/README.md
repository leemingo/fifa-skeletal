# README
Sample code for visualization

## Usage

Requirements:
```
Python 3.10.x
Libraries: numpy, cv2, torch, tqdm
aitviewer (installed via pip install -e ../aitviewer)
```

## Data
Please prepare the data in the following format:

```
<args.data_path> (default: ./data/)
├── cameras_train
│   └── *.npz
├── videos_train
│   └── *.mp4
└── poses
    └── *.npz
└── models
    └── smpl
        └── SMPL_MALE.pkl
```

### Notes for SMPL Model:

1. **Download the SMPL model**: You can download the SMPL model (version 1.1.0 for Python 2.7) from the [official SMPL website](https://smpl.is.tue.mpg.de/).
2. **Rename the downloaded file**: After downloading, rename the file `basicmodel_m_lbs_10_207_0_v1.0.0.pkl` to `SMPL_MALE.pkl`.
3. **Move it to the correct directory**: Place the renamed `SMPL_MALE.pkl` file in the above directory.

## Execution
```bash
python main.py --sequence ARG_FRA_182345 --data_path ./data --output_type both
```

You can also review the image and video generation process in the test.ipynb notebook. 