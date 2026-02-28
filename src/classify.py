import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


class RoadDamageClassifier:
    def __init__(self, model_path='best_model.h5'):
        self.model = load_model(model_path, compile=False)

        self.classes = ['Normal', 'Rusak Berat', 'Rusak Ringan', 'Rusak Sedang']
        self.input_size = (224, 224)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Video tidak bisa dibuka.")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            raise ValueError("FPS tidak terbaca.")

        # Ekstraksi setiap 0.5 detik (sesuai metodologi skripsi)
        interval = int(fps * 0.5)

        frame_count = 0
        total_frames_processed = 0

        # Untuk voting
        sum_predictions = np.zeros(len(self.classes))
        vote_counts = np.zeros(len(self.classes))

        batch_frames = []
        batch_size = 16

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                frame_resized = cv2.resize(frame, self.input_size)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                batch_frames.append(frame_rgb)
                total_frames_processed += 1

                # Jika batch penuh → prediksi
                if len(batch_frames) == batch_size:
                    self._process_batch(batch_frames, sum_predictions, vote_counts)
                    batch_frames = []

            frame_count += 1

        # Proses sisa batch
        if batch_frames:
            self._process_batch(batch_frames, sum_predictions, vote_counts)

        cap.release()

        if total_frames_processed == 0:
            raise ValueError("Tidak ada frame yang diproses.")

        # Soft Voting (rata-rata probabilitas)
        avg_predictions = sum_predictions / total_frames_processed
        class_index = np.argmax(avg_predictions)

        result = {
            "condition": self.classes[class_index],
            "score": round(float(avg_predictions[class_index] * 100), 2),
            "avg_prob": {
                self.classes[i]: round(float(avg_predictions[i] * 100), 2)
                for i in range(len(self.classes))
            },
            "votes": {
                self.classes[i]: int(vote_counts[i])
                for i in range(len(self.classes))
            },
            "total_frames": total_frames_processed
        }

        return result

    def _process_batch(self, batch_frames, sum_predictions, vote_counts):
        batch_array = np.array(batch_frames, dtype=np.float32)
        batch_array = preprocess_input(batch_array)

        predictions = self.model.predict(batch_array, verbose=0)

        for pred in predictions:
            sum_predictions += pred
            vote_counts[np.argmax(pred)] += 1