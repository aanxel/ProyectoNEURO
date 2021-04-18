# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd


DATA_FILE = 'data/data_file.csv'
IM_EXT = 'jpg'
N_ZEROS = 4
IMG_BASE_DIR = 'data/train/'


def load_frame(path, dim=(128, 128), bw=True, norm=True):
    im = Image.open(path)
    if bw:
        im = im.convert('L')
    if dim is not None:
        im = im.resize(dim)
    im = np.array(im)
    if norm:
        im = im / 255
    return im


def load_video(base_path, n_frames, dim=(128, 128), bw=True, norm=True):
    video = []
    for n_frame in range(n_frames):
        frame_path = f'{base_path}-{str(n_frame + 1).zfill(N_ZEROS)}.{IM_EXT}'
        video.append(load_frame(frame_path, dim=dim, bw=bw, norm=norm))
    return np.array(video)


def load_dataset(por_train=0.8, data_file=DATA_FILE, img_base_dir=IMG_BASE_DIR,
                 dim=(128, 128), bw=True, norm=True):
    df = pd.read_csv(data_file, sep=',', header=None).sample(frac=1).to_numpy()
    max_frames = df[:, -1].min()
    X = []
    y = np.unique(df[:, 1], return_inverse=True)[1]
    for frame_base_path in df[1:, 1:3]:
        path = f'{img_base_dir}{frame_base_path[0]}/{frame_base_path[1]}'
        X.append(load_video(base_path=path, n_frames=max_frames, dim=dim,
                            bw=bw, norm=norm))
    X = np.array(X)
    if bw:
        X = X.reshape(X.shape + (1,))
    separador = int(por_train * len(X))
    X_train, X_test = X[:separador], X[separador:]
    y_train, y_test = y[:separador], y[separador:]
    return X_train, X_test, y_train, y_test
