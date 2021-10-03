import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

PROCESS_DIR = ".process"

VIDEOS: dict[str, str] = {
    r"D:\UserData\Videos\Streamed\Process\A.mp4": "亞森 / Arsene",
    r"D:\UserData\Videos\Streamed\Process\B.mp4": "巴哈姆特 / Bahamut",
    r"D:\UserData\Videos\Streamed\Process\R.mp4": "尼德．勇 / R!Nidhogg",
}

HP_RECT: tuple[int, int, int, int] = (212, 77, 712, 82)  # LT-X, LT-Y, RB-X, RB-Y
HP_WIDTH: int = HP_RECT[2] - HP_RECT[0]

REDNESS_THRESHOLD = 50

Color = tuple[int, int, int]  # RGB


def get_video_name(video_path: str) -> str:
    return os.path.basename(video_path)


def export_frames(video_path: str) -> None:
    # Create export directory
    image_dir = os.path.join(PROCESS_DIR, get_video_name(video_path))
    os.makedirs(image_dir, exist_ok=True)

    os.system(f"ffmpeg -i {video_path} -r 1 {image_dir}/%d.png")


def get_redness(color: Color) -> float:
    return color[0] - max(color[1], color[2])


def average_image_color(image: Image) -> tuple[int, int, int]:
    # Obtained from https://gist.github.com/olooney/1246268
    h = image.histogram()

    # split into R, G, B
    r = h[0:256]
    g = h[256:256 * 2]
    b = h[256 * 2: 256 * 3]

    # perform the weighted average of each channel:
    # the *index* is the channel value, and the *value* is its weight
    return (
        sum(i * w for i, w in enumerate(r)) / sum(r),
        sum(i * w for i, w in enumerate(g)) / sum(g),
        sum(i * w for i, w in enumerate(b)) / sum(b)
    )


def is_hp_bar_available(image: Image) -> bool:
    image_hp_bar = image.crop(HP_RECT)

    weights = average_image_color(image_hp_bar)
    max_weight = max(weights)

    return weights.index(max_weight) == 0


def load_hp_of_index(video_name: str, index: int) -> Optional[float]:
    image_path = os.path.join(PROCESS_DIR, video_name, f"{index}.png")

    with Image.open(image_path) as image:
        if not is_hp_bar_available(image):
            return None

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


def sanitize_hp_data(hp_data_raw: list[Optional[float]]) -> list[float]:
    not_none_idx_first = next(idx for idx, hp_data in enumerate(hp_data_raw) if hp_data is not None)
    not_none_idx_last = next(idx for idx, hp_data in reversed(list(enumerate(hp_data_raw))) if hp_data is not None)

    hp_data_stripped = hp_data_raw[not_none_idx_first:not_none_idx_last + 1]

    prev_data = hp_data_stripped[0]
    for idx in range(len(hp_data_stripped)):
        if hp_data := hp_data_stripped[idx]:
            prev_data = hp_data
            continue

        hp_data_stripped[idx] = prev_data

    return hp_data_stripped


def transform_hp_data(hp_data: list[float]) -> list[float]:
    return [hp_data_single * 100 for hp_data_single in hp_data]


def hp_to_dps(hp_data: list[float]) -> list[float]:
    dps_data: list[float] = [0]
    period = 25

    for idx in range(1, len(hp_data)):
        prev_idx = max(0, idx - period)

        dps_data.append((hp_data[prev_idx] - hp_data[idx]) / (idx - prev_idx))

    return dps_data


def plot_data_collection(data_collection: dict[str, list[float]], title: str) -> None:
    sns.set(
        font="Microsoft JhengHei",  # Unicode characters cannot render if font is not set
        rc={"figure.figsize": (12, 8)},
    )
    plot = sns.lineplot(data=data_collection)

    plot.set(
        xlabel="經過秒數 / Sec. Passed",
        ylabel="HP %",
        title=title,
    )

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def main() -> None:
    # for video_path in VIDEOS.keys():
    #     export_frames(video_path)

    all_hp_data: dict[str, list[float]] = {}
    all_dps_data: dict[str, list[float]] = {}

    for video_path, name in VIDEOS.items():
        video_name = get_video_name(video_path)

        hp_data = load_hp_data_raw(video_name)
        hp_data = sanitize_hp_data(hp_data)
        hp_data = transform_hp_data(hp_data)

        all_hp_data[name] = hp_data
        all_dps_data[name] = hp_to_dps(hp_data)

    plot_data_collection(all_hp_data, "阿修羅 超級 P1 (老王) 血線變化 / Master Asura P1 HP as Alberius")
    plot_data_collection(all_dps_data, "阿修羅 超級 P1 (老王) DPS / Master Asura P1 DPS as Alberius")


if __name__ == '__main__':
    main()
