import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="Phone Detector API")

# 1. Configure Gemini API Prefers Environment Variable for safety
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDyIYfOYkOMWlQJqi-WSDzulf0Qys9xemc")
genai.configure(api_key=API_KEY)

# Using Gemini Flash Lite or fallback to standard 1.5 flash
model = genai.GenerativeModel("gemini-2.0-flash-lite-preview-02-05")

class ImagePayload(BaseModel):
    image_base64: str

# Define the exact prompt required
PROMPT = """Analyze image. Return ONLY JSON: {"is_mobile_device_found": bool, "is_person_holding_phone": bool, "bbx_mobile_device": {"x1": ?, "x2": ?, "y1": ?, "y2": ?} or null, "bbx_person_holding_phone": {"x1": ?, "x2": ?, "y1": ?, "y2": ?} or null}. Rules: Coordinates are percentages 0-100. 0,0 is top-left. Optimizations: Strictly distinguish handheld mobile phones from POS screens, desk phones, or calculators. Detect phones held in hands, even if the person is wearing gloves or the phone is small. For the person's bounding box, include their full visible body even if partially blocked by a counter. Detect phones in gloved hands. Precise boxes for small objects. Account for high-angle perspective."""

@app.post("/analyze")
async def analyze(image_payload: ImagePayload):
    try:
        # 2. Format the image data for Gemini
        image_part = {"mime_type": "image/jpeg", "data": image_payload.image_base64}
        # 3. Call Gemini API, strictly enforcing JSON output format
        response = model.generate_content(
            [PROMPT, image_part],
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        # 4. Return the parsed JSON
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Boilerplate to run the app if executed directly
if __name__ == "__main__":
    import uvicorn
    # Railway uses the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
