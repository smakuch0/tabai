from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import logging

from b.TSN import TSN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TSN Guitar Tab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
async def startup():
    global model
    logger.info("Initializing TSN...")
    
    model = TSN()
    model.build_model()
    
    checkpoint_path = os.getenv('MODEL_CHECKPOINT', 'b/checkpoints/best.pth')
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        model.load_weights(checkpoint_path)
        logger.info("Weights loaded")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}, using untrained model")
    
    logger.info("TSN ready")


@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "TSN",
        "model_loaded": model is not None
    }


@app.post("/generate-tab")
async def generate_tab(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    allowed = ['.wav', '.mp3']
    if not any(file.filename.endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail=f"Supported exts: {', '.join(allowed)}")
    
    logger.info(f"Processing: {file.filename}")
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        audio_repr = model.preprocess_audio(tmp_path)
        
        predictions = model.predict(audio_repr, context_window=9)
        
        tab_text = generate_tab_text(predictions)
        
        logger.info(f"Generated {len(predictions)} frames")
        
        return JSONResponse({
            "status": "success",
            "tab": tab_text,
            "frames": len(predictions),
            "filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def generate_tab_text(predictions):
    strings = ['E', 'A', 'D', 'G', 'B', 'e']
    lines = []
    
    for i in range(5, -1, -1):
        line = strings[i] + '|'
        for frame in predictions:
            fret = frame[i]
            if fret == 0:
                line += '-'
            elif fret < 10:
                line += str(fret)
            else:
                line += chr(ord('A') + fret - 10)
        line += '|'
        lines.append(line)
    
    return '\n'.join(lines)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
