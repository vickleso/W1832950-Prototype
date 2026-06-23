from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import Detector, TwHINDetector
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
    # Try to load TwHIN model (lighter, faster)
    twhin_detector = TwHINDetector()
    detector = None  # Keep Qwen as fallback
    print("[INIT] ✓ TwHIN-BERT model loaded successfully")
except Exception as e:
    print(f"[INIT] ✗ Could not load TwHIN model: {e}")
    twhin_detector = None
    try:
        detector = Detector()
        print("[INIT] ✓ Qwen3-VL model loaded as fallback")
    except Exception as e2:
        print(f"[INIT] ✗ Could not load Qwen model: {e2}")
        detector = None

try:
    x_api = XAPIHandler()
except Exception as e:
    print(f"[INIT] ✗ Warning: Could not load X API handler: {e}")
    x_api = None

class AnalyzeRequest(BaseModel):
    url: str

@app.post("/analyse")
async def analyse(request: AnalyzeRequest):
    try:
        if not x_api:
            return {'error': 'X API handler not initialized'}
        
        if not twhin_detector and not detector:
            return {'error': 'No models loaded (TwHIN and Qwen unavailable)'}
        
        print(f"\n{'='*60}")
        print(f"Processing: {request.url}")
        print(f"{'='*60}")
        
        # Fetch post data from X API
        post = x_api.analyze_url(request.url)
        
        # Validate fetched data
        if not post:
            return {'error': 'Failed to fetch post data'}
        
        print("\n[MAIN] Post data received:")
        print(f"  - Author: {post.get('author', 'N/A')}")
        print(f"  - Text: {post.get('text', 'N/A')[:100]}...")
        print(f"  - Media URLs: {post.get('media_urls', [])}")
        
        # Validate text
        if not post.get('text'):
            return {'error': 'Tweet has no text content'}
        
        # Use TwHIN for fast, accurate text classification
        if twhin_detector:
            print(f"\n[MAIN] Using TwHIN-BERT for analysis...")
            result = twhin_detector.analyse(post['text'])
            model_used = "TwHIN-BERT"
        else:
            # Fallback to Qwen if TwHIN unavailable
            print(f"\n[MAIN] TwHIN unavailable, using Qwen3-VL for analysis...")
            image_url = post['media_urls'][0] if post.get('media_urls') else None
            result = detector.analyse(post['text'], image_url)
            model_used = "Qwen3-VL"
        
        print("[MAIN] Model response:")
        print(f"  - Classification: {result['classification']}")
        print(f"  - Confidence: {result['confidence']}\n")
        
        # Return results
        response = {
            'url': request.url,
            'author': post.get('author', 'Unknown'),
            'text': post.get('text', ''),
            'classification': result['classification'],
            'confidence': result['confidence'],
            'model': model_used,
            'likes': post.get('metrics', {}).get('like_count', 0),
            'retweets': post.get('metrics', {}).get('retweet_count', 0)
        }
        
        # Add details if available
        if 'details' in result:
            response['details'] = result['details']
        
        print(f"[MAIN] Returning response: {response}\n")
        return response
        
    except Exception as e:
        print(f"\n✗ Error in /analyse: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

@app.get("/health")
async def health():
    models_loaded = []
    if twhin_detector:
        models_loaded.append("TwHIN-BERT")
    if detector:
        models_loaded.append("Qwen3-VL")
    
    return {
        'status': 'ok',
        'models': models_loaded,
        'primary_model': models_loaded[0] if models_loaded else 'none'
    }
