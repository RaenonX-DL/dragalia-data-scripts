import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image, ImageDraw
import pytesseract
import matplotlib.pyplot as plt
import seaborn as sns

PROCESS_DIR = ".process"


@dataclass
class Video:
    path: str
    label: str
    total_damage_override: Optional[float] = None
    hp_mapping_fn: Optional[Callable[[float], float]] = None

    name: str = field(init=False)
    total_damage: Optional[float] = field(init=False)

    def __post_init__(self):
        self.name = os.path.basename(self.path)
        self.total_damage = int(
            self.hp_mapping_fn(self.total_damage_override)
            if self.total_damage_override and self.hp_mapping_fn
            else (self.total_damage_override or 0)
        )


VIDEOS: list[Video] = [
    Video(
        path=r"D:\UserData\Videos\Streamed\Process\.PharesDPS.mp4-clip.mp4-crop.mp4",
        label="二哥", total_damage_override=6256474,
    ),
    Video(
        path=r"D:\UserData\Videos\Streamed\Process\.PharesDPSNotte.mp4-clip.mp4-crop.mp4",
        label="那姆", total_damage_override=7674569,
    ),
]

# VIDEOS: list[Video] = [
#     Video(
#         path=r"D:\UserData\Videos\Streamed\Process\.ValyxD.mp4-clip.mp4-crop.mp4",
#         label="四哥 - 變龍", total_damage_override=7748694,
#     ),
#     Video(
#         path=r"D:\UserData\Videos\Streamed\Process\.ValyxGAudricChronos.mp4-clip.mp4-crop.mp4",
#         label="老爸 - 變龍", total_damage_override=8736042,
#     ),
#     Video(
#         path=r"D:\UserData\Videos\Streamed\Process\.ValyxGAudricNoD.mp4-clip.mp4-crop.mp4",
#         label="老爸 - 不龍", total_damage_override=5925266,
#     ),
#     Video(
#         path=r"D:\UserData\Videos\Streamed\Process\.ValyxGLucaChronos.mp4-clip.mp4-crop.mp4",
#         label="琉卡刀 - 變龍", total_damage_override=9159795,
#     ),
# ]

DPS_START = 10
DPS_END = 170

HP_RECT: tuple[int, int, int, int] = (743, 184, 972, 229)  # LT-X, LT-Y, RB-X, RB-Y

HP_RECT_RESULT: tuple[int, int, int, int] = (483, 844, 932, 912)  # LT-X, LT-Y, RB-X, RB-Y

TIMER_RECT: tuple[int, int, int, int] = (841, 44, 888, 75)  # LT-X, LT-Y, RB-X, RB-Y


@dataclass
class DataOnFrame:
    hp: Optional[float]
    timer_sec: Optional[float]


def export_frames(video: Video) -> None:
    # Create export directory
    image_dir = os.path.join(PROCESS_DIR, video.name)

    if os.path.exists(image_dir):
        return

    os.makedirs(image_dir, exist_ok=True)

    # Frames for each second
    os.system(f"ffmpeg -i {video.path} -r 1 {os.path.join(image_dir, '%d.png')}")
    # Last frame
    os.system(f"ffmpeg -sseof -3 -i {video.path} -update 1 -q:v 1 {os.path.join(image_dir, 'last.png')}")


def ocr_num(image: Image, box: tuple[int, int, int, int], config: str = "") -> Optional[float]:
    recog_hp_str = pytesseract.image_to_string(image.crop(box), config=config)

    try:
        return float("".join([s for s in recog_hp_str if s.isdigit()]))
    except Exception:
        return None


def load_hp_of_index(image_path: str) -> Optional[float]:
    with Image.open(image_path) as image:
        return ocr_num(image, HP_RECT)


def load_data_of_frame(video: Video, index: int) -> DataOnFrame:
    image_path = os.path.join(PROCESS_DIR, video.name, f"{index}.png")

    with Image.open(image_path) as image:
        hp = ocr_num(image, HP_RECT)
        timer_sec = ocr_num(image, TIMER_RECT, config="--psm 7")

    if hp and video.hp_mapping_fn:
        hp = video.hp_mapping_fn(hp)

    return DataOnFrame(hp=hp, timer_sec=timer_sec)


def load_frame_data(video: Video) -> list[DataOnFrame]:
    _, _, files = next(os.walk(os.path.join(PROCESS_DIR, video.name)))
    file_count = len(files) - 1  # 1 for `last.png` which is the last frame

    print(f"{file_count} HP data for {video.name} to process.")

    ret: list[DataOnFrame] = []

    chunk_size = 40

    for start_idx in range(1, file_count + 1, chunk_size):
        print(f"Processing {start_idx} / {file_count} HP data... ({video.name})")
        with ThreadPoolExecutor() as executor:
            futures = []

            for idx in range(start_idx, min(start_idx + chunk_size, file_count + 1)):
                futures.append(executor.submit(load_data_of_frame, video, idx))

            ret.extend(future.result() for future in futures)

    return ret


def sanitize_frame_data(frame_data: list[DataOnFrame]) -> list[DataOnFrame]:
    not_none_idx_first = next(
        idx for idx, frame_data_single in enumerate(frame_data)
        if frame_data_single.hp is not None
    )
    not_none_idx_last = next(
        idx for idx, frame_data_single in reversed(list(enumerate(frame_data)))
        if frame_data_single.hp is not None
    )

    # Strip `None` of head nad tail
    data_stripped = frame_data[not_none_idx_first:not_none_idx_last + 1]

    # Remove data where the timer is the same
    data_ret: list[DataOnFrame] = []
    prev_timer = -1
    for data in data_stripped:
        if data.timer_sec is not None and data.timer_sec == prev_timer:
            continue

        prev_timer = data.timer_sec
        data_ret.append(data)

    return data_ret


def smooth_frame_data(frame_data: list[DataOnFrame]) -> list[DataOnFrame]:
    last_max = frame_data[0].hp
    prev_outlier = False

    # Ensure decreasing
    for idx in range(1, len(frame_data)):
        if not frame_data[idx].hp:
            continue  # Current HP data could be `None`

        if frame_data[idx].hp < last_max:
            frame_data[idx].hp = None
            print(f"HP Decreasing at index #{idx}", file=sys.stderr)
            continue

        if abs(last_max - frame_data[idx].hp) > 1000000:
            print(
                f"Large HP data gap found at index #{idx}: {abs(last_max - frame_data[idx].hp):5.3f}",
                file=sys.stderr
            )

            if prev_outlier:
                last_max = frame_data[idx].hp

            prev_outlier = not prev_outlier
            frame_data[idx].hp = None
            continue  # Potential outlier data

        prev_outlier = False
        frame_data[idx].hp = last_max = max(last_max, frame_data[idx].hp)

    return frame_data


def fill_gap(data: list[DataOnFrame]) -> list[DataOnFrame]:
    data = data.copy()

    prev_none_idx = None
    for idx in range(len(data)):
        if current_hp := data[idx].hp:
            # Data point available
            if not prev_none_idx:
                continue

            if idx - prev_none_idx > 3:
                print(f"Empty data points # > 3 ({idx - prev_none_idx}): "
                      f"{prev_none_idx} ~ {idx - 1}", file=sys.stderr)

            prev_available_idx = prev_none_idx - 1
            if prev_none_idx > 0:
                prev_available_hp = data[prev_available_idx].hp
            else:
                prev_available_hp = 1

            step = (prev_available_hp - current_hp) / (idx - prev_available_idx)
            for step_count, idx_fill in enumerate(range(prev_none_idx, idx), start=1):
                data[idx_fill].hp = prev_available_hp - step * step_count

            prev_none_idx = None
            continue

        # Data point unavailable
        if prev_none_idx is None:  # First unavailable data point, record it
            prev_none_idx = idx

    while not data[-1]:
        data.pop(-1)

    return data


def frame_data_hp_to_k(data: list[DataOnFrame]) -> None:
    for single_data in data:
        if not single_data.hp:
            continue

        single_data.hp /= 1000


def hp_to_dps(frame_data: list[DataOnFrame]) -> list[float]:
    frame_data = fill_gap(frame_data)
    dps_data: list[float] = [0]
    period = 25

    for idx in range(1, len(frame_data)):
        prev_idx = max(0, idx - period)

        dps_data.append((frame_data[idx].hp - frame_data[prev_idx].hp) / (idx - prev_idx))

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
        ylabel="HP (K)",
        title=title,
    )

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def show_final_hp_result(video: Video):
    image_path = os.path.join(PROCESS_DIR, video.name, f"last.png")

    with Image.open(image_path) as image:
        image = image.crop(HP_RECT_RESULT)
        draw = ImageDraw.Draw(image)
        draw.text((5, 5), video.name, (255, 255, 255))
        plt.imshow(image)
        plt.show()


def main() -> None:
    for video in VIDEOS:
        print(f"Exporting frames of {video.path}...")
        export_frames(video)

    all_damage_data: dict[str, list[DataOnFrame]] = {}
    all_dps_data: dict[str, list[float]] = {}

    for video in VIDEOS:
        frame_data = load_frame_data(video)
        frame_data = sanitize_frame_data(frame_data)
        frame_data = smooth_frame_data(frame_data)
        # May have starting or trailing outlier to be removed
        frame_data = sanitize_frame_data(frame_data)

        name_key = f"{video.label} (Damage: {video.total_damage or 'N/A'})"

        frame_data_hp_to_k(frame_data)

        all_damage_data[name_key] = frame_data
        all_dps_data[name_key] = hp_to_dps(frame_data)

        show_final_hp_result(video)

    for name, frame_data in all_damage_data.items():
        frame = all_dps_data[name][DPS_START:DPS_END]

        print(f"{name}: DPS avg ({DPS_START} ~ {DPS_END}) {sum(frame) / len(frame):.4f} K / s")

    plot_data_collection(all_dps_data, "DPS")
    plot_data_collection(
        {
            name: [frame_data_single.hp for frame_data_single in frame_data]
            for name, frame_data in all_damage_data.items()
        },
        "總傷害 / Total Damage"
    )


if __name__ == '__main__':
    main()
