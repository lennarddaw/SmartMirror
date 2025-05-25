import os
import cv2
import csv


class VideoSegmenter:
    """
    A class for labeling rep starts in videos. Labels are binary: 1.0 for frames where the user presses SPACE, 0.0 otherwise.
    """

    def __init__(self,
                 videos_dir: str = "videos",
                 labels_dir: str = "labeled_videos",
                 csv_header: list = None,
                 label_key: int = 32,
                 stop_key: int = 27,
                 wait_ms: int = 100):
        """
        Args:
            videos_dir: Directory containing .mp4 video files.
            labels_dir: Directory where CSV label files will be saved.
            csv_header: Header row for CSV files. Defaults to ["frame_index", "label"].
            label_key: Keycode to register a label (default SPACE).
            stop_key: Keycode to stop labeling (default ESC).
            wait_ms: Delay per frame in milliseconds for display.
        """
        self.videos_dir = videos_dir
        self.labels_dir = labels_dir
        os.makedirs(self.labels_dir, exist_ok=True)

        self.csv_header = csv_header or ["frame_index", "label"]
        self.label_key = label_key
        self.stop_key = stop_key
        self.wait_ms = wait_ms

    def label_reps(self, video_filename: str) -> bool:
        """
        Interactively label rep start frames in a single video.
        Press SPACE to mark a rep (label=1), ESC to stop.

        Args:
            video_filename: Name of the .mp4 file in videos_dir.
        Returns:
            True if completed, False if aborted.
        """
        video_path = os.path.join(self.videos_dir, video_filename)
        csv_path = os.path.join(
            self.labels_dir,
            os.path.splitext(video_filename)[0] + ".csv"
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        labels = [0.0] * total_frames

        print(f"Labeling {video_filename}. Press SPACE to label, ESC to finish.")

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(
                frame,
                f"Frame: {idx}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            cv2.imshow("Label Reps", frame)
            key = cv2.waitKey(self.wait_ms)

            if key == self.label_key:
                labels[idx] = 1.0
                print(f"Labeled frame {idx}: 1.0")
            elif key == self.stop_key:
                print("Labeling stopped by user.")
                break

            idx += 1
            if idx >= total_frames:
                break

        cap.release()
        cv2.destroyAllWindows()

        self._save_labels(csv_path, labels)
        print(f"Labels saved to {csv_path}")
        return True

    def _save_labels(self, csv_path: str, labels: list):
        """Save labels list to CSV file with header."""
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)
            for i, lbl in enumerate(labels):
                writer.writerow([i, lbl])

    def get_labeled_frames(self, video_filename: str):
        """
        Load frames and their labels from a video and its CSV file.

        Args:
            video_filename: Name of the .mp4 file in videos_dir.
        Returns:
            List of tuples (label: float, frame: ndarray).
        """
        video_path = os.path.join(self.videos_dir, video_filename)
        csv_path = os.path.join(
            self.labels_dir,
            os.path.splitext(video_filename)[0] + ".csv"
        )

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Labels not found: {csv_path}")

        labels = []
        with open(csv_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(float(row["label"]))

        cap = cv2.VideoCapture(video_path)
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or idx >= len(labels):
                break
            frames.append((labels[idx], frame))
            idx += 1
        cap.release()
        return frames

    def batch_label_all(self):
        """
        Label all .mp4 videos in videos_dir that do not yet have a corresponding CSV file.
        """
        files = [f for f in os.listdir(self.videos_dir) if f.lower().endswith(".mp4")]
        labeled = {os.path.splitext(f)[0] for f in os.listdir(self.labels_dir) if f.lower().endswith(".csv")}

        for video in files:
            base = os.path.splitext(video)[0]
            if base in labeled:
                print(f"Already labeled: {video}")
                continue
            success = self.label_reps(video)
            if not success:
                print("Batch labeling aborted.")
                break


if __name__ == "__main__":
    segmenter = VideoSegmenter()
    segmenter.batch_label_all()
