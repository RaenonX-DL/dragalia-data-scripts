import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

PROCESS_DIR = ".process"

BOSS_NAME_CHT: str = "風寶龍 70F"

BOSS_NAME_EN: str = "Wind MG 70F"

VIDEOS: dict[str, str] = {
    r"D:\UserData\Videos\Streamed\Process\.AO.MG.AO.mp4-clip.mp4-crop.mp4": "火牙 (瑪斯) / A&O (Mars)",
    r"D:\UserData\Videos\Streamed\Process\.AO.MG.AO.Agni.mp4-clip.mp4-crop.mp4": "火牙 (R70) / A&O (Agni)",
    r"D:\UserData\Videos\Streamed\Process\.AO.MG.Gatov.mp4-clip.mp4-crop.mp4": "葛托夫 / Gatov",
}

MAX_HP_DATA_GAP = 0.1

DPS_START = 10
DPS_END = 170

HP_RECT: tuple[int, int, int, int] = (212, 77, 712, 82)  # LT-X, LT-Y, RB-X, RB-Y
HP_WIDTH: int = HP_RECT[2] - HP_RECT[0]

REDNESS_THRESHOLD = 185

Color = tuple[int, int, int]  # RGB


def get_video_name(video_path: str) -> str:
    return os.path.basename(video_path)


def export_frames(video_path: str) -> None:
    # Create export directory
    image_dir = os.path.join(PROCESS_DIR, get_video_name(video_path))

    if os.path.exists(image_dir):
        return

    os.makedirs(image_dir, exist_ok=True)

    os.system(f"ffmpeg -i {video_path} -r 1 {image_dir}/%d.png")


def get_redness(color: Color) -> float:
    return color[0] - max(color[1], color[2])


def get_highest_channel_index(image: Image) -> int:
    # Obtained from https://gist.github.com/olooney/1246268
    h = image.histogram()

    # split into R, G, B
    r = h[0:256]
    g = h[256:256 * 2]
    b = h[256 * 2: 256 * 3]

    # perform the weighted average of each channel:
    # the *index* is the channel value, and the *value* is its weight
    weights = (
        sum(i * w for i, w in enumerate(r)) / sum(r),
        sum(i * w for i, w in enumerate(g)) / sum(g),
        sum(i * w for i, w in enumerate(b)) / sum(b)
    )

    return weights.index(max(weights))


def is_hp_bar_available(image: Image) -> bool:
    image_hp_bar = image.crop(HP_RECT)

    return get_highest_channel_index(image_hp_bar) == 0


def load_hp_of_index(video_name: str, index: int) -> Optional[float]:
    image_path = os.path.join(PROCESS_DIR, video_name, f"{index}.png")

    with Image.open(image_path) as image:
        for x in range(HP_RECT[2], HP_RECT[0] - 1, -1):
            if get_redness(image.getpixel((x, HP_RECT[3]))) < REDNESS_THRESHOLD:
                continue

            return (x + 1 - HP_RECT[0]) / HP_WIDTH

    return None


def load_hp_data_raw(video_name: str) -> list[Optional[float]]:
    _, _, files = next(os.walk(os.path.join(PROCESS_DIR, video_name)))
    file_count = len(files)

    print(f"{file_count} HP data for {video_name} to process.")

    ret: list[float] = []

    chunk_size = 40

    for start_idx in range(1, file_count + 1, chunk_size):
        print(f"Processing {start_idx} / {file_count} HP data... ({video_name})")
        with ThreadPoolExecutor() as executor:
            futures = []

            for idx in range(start_idx, min(start_idx + chunk_size, file_count + 1)):
                futures.append(executor.submit(load_hp_of_index, video_name, idx))

            ret.extend(future.result() for future in futures)

    return ret


def sanitize_hp_data(hp_data_raw: list[Optional[float]]) -> list[Optional[float]]:
    not_none_idx_first = next(
        idx for idx, hp_data in enumerate(hp_data_raw)
        if hp_data is not None
    )
    not_none_idx_last = next(
        idx for idx, hp_data in reversed(list(enumerate(hp_data_raw)))
        if hp_data is not None
    )

    # Strip `None` of head nad tail
    hp_data_stripped = hp_data_raw[not_none_idx_first:not_none_idx_last + 1]

    return hp_data_stripped


def smooth_hp_data(hp_data: list[Optional[float]]) -> list[Optional[float]]:
    last_min = hp_data[0]
    prev_outlier = False

    # Ensure decreasing
    for idx in range(1, len(hp_data)):
        if not hp_data[idx]:
            continue  # Current data could be `None`

        if abs(last_min - hp_data[idx]) > MAX_HP_DATA_GAP:
            print(
                f"Large HP data gap found at index #{idx:>3}: {abs(last_min - hp_data[idx]):5.3f}",
                file=sys.stderr
            )

            if prev_outlier:
                last_min = hp_data[idx]

            prev_outlier = not prev_outlier
            hp_data[idx] = None
            continue  # Potential outlier data

        prev_outlier = False
        hp_data[idx] = last_min = min(last_min, hp_data[idx])

    return hp_data


def fill_gap(data: list[Optional[float]]) -> list[float]:
    data = data.copy()

    prev_none_idx = None
    for idx in range(len(data)):
        if current_hp := data[idx]:
            # Data point available
            if not prev_none_idx:
                continue

            if idx - prev_none_idx > 3:
                print(f"Empty data points # > 3 ({idx - prev_none_idx}): "
                      f"{prev_none_idx} ~ {idx - 1}", file=sys.stderr)

            prev_available_idx = prev_none_idx - 1
            if prev_none_idx > 0:
                prev_available_hp = data[prev_available_idx]
            else:
                prev_available_hp = 1

            step = (prev_available_hp - current_hp) / (idx - prev_available_idx)
            for step_count, idx_fill in enumerate(range(prev_none_idx, idx), start=1):
                data[idx_fill] = prev_available_hp - step * step_count

            prev_none_idx = None
            continue

        # Data point unavailable
        if prev_none_idx is None:  # First unavailable data point, record it
            prev_none_idx = idx

    while not data[-1]:
        data.pop(-1)

    return data


def data_times_100(data: list[Optional[float]]) -> list[Optional[float]]:
    return [data_single * 100 if data_single else data_single for data_single in data]


def hp_to_dps(hp_data: list[float]) -> list[float]:
    hp_data = fill_gap(hp_data)
    dps_data: list[float] = [0]
    period = 25

    for idx in range(1, len(hp_data)):
        prev_idx = max(0, idx - period)

        dps_data.append((hp_data[prev_idx] - hp_data[idx]) / (idx - prev_idx))

    return dps_data


def plot_data_collection(
        data_collection: dict[str, list[float]], title: str,
        show_marker: bool = False,
) -> None:
    sns.set(
        font="Microsoft JhengHei",  # Unicode characters cannot render if font is not set
        rc={"figure.figsize": (12, 8)},
    )
    plot = sns.lineplot(data=data_collection, marker="o" if show_marker else None)

    plot.set(
        xlabel="經過秒數 / Sec. Passed",
        ylabel="HP %",
        title=title,
    )

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def main() -> None:
    for video_path in VIDEOS.keys():
        print(f"Exporting frames of {video_path}...")
        export_frames(video_path)

    all_hp_data: dict[str, list[Optional[float]]] = {}
    all_dps_data: dict[str, list[float]] = {}

    for video_path, name in VIDEOS.items():
        video_name = get_video_name(video_path)

        hp_data = load_hp_data_raw(video_name)
        hp_data = sanitize_hp_data(hp_data)
        hp_data = smooth_hp_data(hp_data)
        hp_data = data_times_100(hp_data)

        all_hp_data[name] = hp_data
        all_dps_data[name] = hp_to_dps(hp_data)

    for name, hp_data in all_hp_data.items():
        print(f"{name}: Ends at {hp_data[-1]}")

        frame = all_dps_data[name][DPS_START:DPS_END]

        print(f"{name}: DPS avg ({DPS_START} ~ {DPS_END}) {sum(frame) / len(frame):.4f}")

    plot_data_collection(
        all_dps_data,
        f"{BOSS_NAME_CHT} DPS / {BOSS_NAME_EN} DPS"
    )
    plot_data_collection(
        all_hp_data,
        f"{BOSS_NAME_CHT} 血量變化 / {BOSS_NAME_EN} HP"
    )


if __name__ == '__main__':
    main()
