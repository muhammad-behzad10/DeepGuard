"""DeepGuard laptop demo: temporary upload -> validated Xception pilot -> result."""
from contextlib import asynccontextmanager
from hashlib import sha256
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from uuid import uuid4
from fastapi import FastAPI, HTTPException, UploadFile
from api_contract import AnalysisReceipt, VideoInputError
from video_validation import VideoValidationError, validate_video

MAX_VIDEO_BYTES = 30_000_000
CHUNK_BYTES = 1024 * 1024
API_VERSION = '0.6.0'
logger = logging.getLogger('uvicorn.error')

def load_engine():
    from inference_service import XceptionEngine
    return XceptionEngine()

def create_app(engine_factory=load_engine) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = None
        try:
            app.state.engine = engine_factory()
            logger.info('DeepGuard: verified Xception and YuNet ready on CPU.')
        except Exception:
            logger.exception('DeepGuard model unavailable. Check .venv-model, pilot files and model hashes.')
        yield
        app.state.engine = None

    app = FastAPI(title='DeepGuard API', version=API_VERSION, lifespan=lifespan,
                  description='Research prototype for facial manipulation assessment.')
    app.state.engine = None
    app.state.analysis_lock = Lock()

    @app.get('/')
    def home():
        return {'message': 'DeepGuard backend is running', 'api_version': API_VERSION}

    @app.get('/health')
    def health_check():
        return {'status': 'ok', 'service': 'DeepGuard', 'api_version': API_VERSION,
                'model_loaded': app.state.engine is not None, 'busy': app.state.analysis_lock.locked()}

    @app.post('/upload-video', response_model=AnalysisReceipt)
    def upload_video(file: UploadFile):
        locked = False
        try:
            filename = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
            if not filename.lower().endswith('.mp4'):
                raise HTTPException(415, 'Please upload an MP4 file.')
            if file.size is not None and file.size > MAX_VIDEO_BYTES:
                raise HTTPException(413, 'The video exceeds the 30 MB limit.')
            engine = app.state.engine
            if engine is None:
                raise HTTPException(503, 'The analysis model is not ready. Check the backend setup and restart it.')
            locked = app.state.analysis_lock.acquire(blocking=False)
            if not locked:
                raise HTTPException(429, 'The service is analysing another video. Wait for it to finish, then retry.', headers={'Retry-After': '5'})
            started = perf_counter()
            total_bytes, file_hash = 0, sha256()
            with TemporaryDirectory(prefix='deepguard_') as work_dir:
                video_path = Path(work_dir) / 'video.mp4'
                with video_path.open('wb') as destination:
                    while chunk := file.file.read(CHUNK_BYTES):
                        total_bytes += len(chunk)
                        if total_bytes > MAX_VIDEO_BYTES:
                            raise HTTPException(413, 'The video exceeds the 30 MB limit.')
                        destination.write(chunk)
                        file_hash.update(chunk)
                if total_bytes == 0:
                    raise HTTPException(400, 'The uploaded file is empty.')
                video_info = validate_video(video_path)
                # Align directly from original video using the tested 256x256 RGB pilot.
                result = engine.analyse(video_path)
                receipt = AnalysisReceipt(api_version=API_VERSION, upload_id=uuid4().hex,
                    filename=filename, size_bytes=total_bytes, sha256=file_hash.hexdigest(),
                    stored=False, video=video_info, **result)
            # Temporary files are gone BEFORE returning success.
            receipt.timing_seconds['server_request'] = perf_counter() - started
            return receipt
        except HTTPException:
            raise
        except (VideoValidationError, VideoInputError) as error:
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            logger.exception('DeepGuard video analysis failed.')
            raise HTTPException(500, 'The analysis could not be completed. No prediction was produced.') from error
        finally:
            try:
                file.file.close()
            finally:
                if locked:
                    app.state.analysis_lock.release()
    return app

app = create_app()
