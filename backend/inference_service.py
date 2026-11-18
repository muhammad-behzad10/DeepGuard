"""Use the SAME validated pilot preprocessing, model and score aggregation."""
from pathlib import Path
from time import perf_counter
import cv2
import torch
from api_contract import VideoInputError
from xception_pilot.predict_video import aggregate_prediction
from xception_pilot.video_preprocessing import (
    MIN_FACES, PREPROCESSING_VERSION, SIZE, TARGET_FRAMES,
    make_detector, preprocess_video, rgb_normalized_chw,
)
from xception_pilot.xception_loader import load_xception

class XceptionEngine:
    """One CPU classifier/detector; the API serializes access with its job lock."""
    def __init__(self, model_dir: Path | None = None):
        model_dir = model_dir or Path(__file__).resolve().parent / 'models'
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        self.model, self.metadata = load_xception(model_dir / 'xception_best.pth')
        self.detector = make_detector(model_dir / 'face_detection_yunet_2026may.onnx')

    def analyse(self, path: Path) -> dict:
        start = perf_counter()
        try:
            crops, rows, _, video = preprocess_video(path, self.detector)
        except (ValueError, cv2.error) as error:
            raise VideoInputError('The video could not be prepared. Choose another MP4 with a clearly visible face.') from error
        prepared = perf_counter()
        scores, frame_scores = [], []
        if len(crops) >= MIN_FACES:
            with torch.inference_mode():
                for row in rows:
                    if 'crop_index' not in row:
                        continue
                    crop = crops[row['crop_index']]
                    tensor = torch.from_numpy(rgb_normalized_chw(crop)).unsqueeze(0)
                    logits = self.model(tensor)
                    if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all().item():
                        raise RuntimeError('The model returned invalid scores.')
                    score = float(torch.softmax(logits, dim=1)[0, 1].item())
                    scores.append(score)
                    frame_scores.append({'frame_index': row['frame_index'], 'fake_score': score})
        prediction = aggregate_prediction(scores, threshold=0.5)
        finished = perf_counter()
        return {
            'status': 'analysed' if prediction is not None else 'insufficient_evidence',
            'analysis_performed': prediction is not None,
            'prediction': prediction,
            'reason': None if prediction is not None else (
                f'Only {len(crops)} of {len(rows)} sampled frames provided a usable single face; '
                f'at least {MIN_FACES} are required. Choose a clip with one clearly visible face.'),
            'model': {'id': self.metadata['model_id'], 'checkpoint_sha256': self.metadata['checkpoint_sha256'], 'device': 'cpu'},
            'preprocessing': {
                'version': PREPROCESSING_VERSION, 'sampling_method': video['sampling_method'],
                'target_frames': TARGET_FRAMES, 'sampled_frames': len(rows), 'usable_faces': len(crops),
                'minimum_usable_faces': MIN_FACES, 'requested_frame_indices': video['requested_frame_indices'],
                'counts': video['counts'], 'crop_size': [SIZE, SIZE], 'selection_policy': 'single_face_only',
                'face_tracking_performed': False, 'frames_retained': False, 'crops_retained': False, 'full_decode_checked': False,
            },
            'frame_scores': frame_scores,
            'timing_seconds': {'preprocessing': prepared-start, 'inference': finished-prepared, 'processing': finished-start},
        }
