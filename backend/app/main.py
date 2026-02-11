from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import Detector
from x_api_handler import XAPIHandler

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    detector = Detector()
    x_api = XAPIHandler()
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    detector = None
    x_api = None

class AnalyzeRequest(BaseModel):
    url: str

@app.post("/analyse")
async def analyse(request: AnalyzeRequest):
    try:
        if not x_api or not detector:
            return {'error': 'Models not loaded'}
        
        print(f"\n{'='*60}")
        print(f"Processing: {request.url}")
        print(f"{'='*60}")
        
        # Fetch post data from X API
        post = x_api.analyze_url(request.url)
        
        # Validate fetched data
        if not post:
            return {'error': 'Failed to fetch post data'}
        
        print("\n[MAIN] Post data received:")
        print("  - Author: {post.get('author', 'N/A')}")
        print("  - Text: {post.get('text', 'N/A')[:100]}...")
        print("  - Media URLs: {post.get('media_urls', [])}")
        
        # Validate text
        if not post.get('text'):
            return {'error': 'Tweet has no text content'}
        
        # Extract image URL (use first media if available)
        image_url = post['media_urls'][0] if post.get('media_urls') else None
        print(f"  - Using image: {image_url if image_url else 'None (using placeholder)'}\n")
        
        # Send to model for analysis
        print(f"[MAIN] Sending to model for analysis...")
        result = detector.analyse(post['text'], image_url)
        
        print("[MAIN] Model response:")
        print("  - Classification: {result['classification']}")
        print("  - Confidence: {result['confidence']}")
        print("  - Raw: {result.get('raw', 'N/A')[:100]}...\n")
        
        # Return results
        response = {
            'url': request.url,
            'author': post.get('author', 'Unknown'),
            'text': post.get('text', ''),
            'classification': result['classification'],
            'confidence': result['confidence'],
            'likes': post.get('metrics', {}).get('like_count', 0),
            'retweets': post.get('metrics', {}).get('retweet_count', 0)
        }
        
        print(f"[MAIN] Returning response: {response}\n")
        return response
        
    except Exception as e:
        print(f"\n✗ Error in /analyse: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

@app.get("/health")
async def health():
    return {'status': 'ok', 'model': 'Qwen3-VL-2B-Thinking'}
