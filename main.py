import asyncio
import time
from datetime import datetime
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(
    title="Guard API",
    description="Text detection API using Guard 0.6B model",
    version="1.0.0",
)

# Load model and tokenizer from local directory
local_model_path = "./guard-0.6b"
print("Loading model and tokenizer from local directory...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)
print("Model loaded successfully!")

def detect_text(text: str) -> dict:
    """Run text detection with optimized inference settings."""
    if not text or not text.strip():
        return {"model_response": "Error: Empty input", "time_cost": 0.0}

    messages = [
        {"role": "user", "content": text.strip()},
    ]
    start_time = time.time()

    # Tokenize and generate
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])

    end_time = time.time()
    time_cost = end_time - start_time

    return {"model_response": response, "time_cost": round(time_cost, 4)}

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "model": "guard-0.6b",
            "device": str(model.device),
        }
    )

@app.post("/detect")
async def detect(text: str = Body(..., media_type="text/plain")):
    """Detect endpoint that accepts plain text input.

    - **text**: Plain text content to detect (Content-Type: text/plain)
    """
    try:
        if not text or not text.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Empty request body",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        result = await asyncio.to_thread(detect_text, text)
        return JSONResponse(
            content={
                "success": True,
                "data": result,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
