from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import Detector, TwHINDetector, QwenVLDetector
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
    qwen_detector = None
    print("[INIT] ✓ TwHIN-BERT model loaded successfully")
except Exception as e:
    print(f"[INIT] ✗ Could not load TwHIN model: {e}")
    twhin_detector = None
    try:
        qwen_detector = QwenVLDetector()
        print("[INIT] ✓ Qwen3-VL fine-tuned model loaded")
    except Exception as e2:
        print(f"[INIT] ✗ Could not load Qwen model: {e2}")
        qwen_detector = None

# Fallback for older Qwen loader if needed
if not qwen_detector:
    try:
        detector = Detector()
        qwen_detector = detector
        print("[INIT] ✓ Qwen3-VL fallback model loaded")
    except Exception as e3:
        print(f"[INIT] ✗ Could not load Qwen fallback model: {e3}")
        detector = None
        qwen_detector = None
else:
    detector = qwen_detector

try:
    x_api = XAPIHandler()
except Exception as e:
    print(f"[INIT] ✗ Warning: Could not load X API handler: {e}")
    x_api = None

class AnalyzeRequest(BaseModel):
    # This request schema defines the payload the web UI sends when a user submits a post URL for analysis.
    url: str
    model: str | None = None

@app.post("/analyse")
async def analyse(request: AnalyzeRequest):
    # This endpoint coordinates the full analysis flow from X post retrieval to model inference and response formatting.
    try:
        if not x_api:
            return {'error': 'X API handler not initialized'}
        
        if not twhin_detector and not qwen_detector:
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
        
        # Determine which model to use
        requested = (request.model or '').strip().lower()
        if requested == 'qwen' and qwen_detector:
            print(f"\n[MAIN] Using Qwen3-VL as requested by client...")
            image_url = post['media_urls'][0] if post.get('media_urls') else None #The use image analysis, only done by Qwen
            result = qwen_detector.analyse(post['text'], image_url)
            model_used = "Qwen3-VL"
        elif requested == 'twhin' and twhin_detector:
            print(f"\n[MAIN] Using TwHIN-BERT as requested by client...")
            result = twhin_detector.analyse(post['text'])
            model_used = "TwHIN-BERT"
        elif twhin_detector:
            print(f"\n[MAIN] Using TwHIN-BERT for analysis...")
            result = twhin_detector.analyse(post['text'])
            model_used = "TwHIN-BERT"
        elif qwen_detector:
            print(f"\n[MAIN] TwHIN unavailable, using Qwen3-VL for analysis...")
            image_url = post['media_urls'][0] if post.get('media_urls') else None
            result = qwen_detector.analyse(post['text'], image_url)
            model_used = "Qwen3-VL"
        else:
            return {'error': 'Requested model not available'}
        
        print("[MAIN] Model response:")
        print(f"  - Classification: {result['classification']}")
        print(f"  - Confidence: {result['confidence']}\n")
        
        # Return results
        rag_used = result.get('status') == 'unsure'
        response = {
            'url': request.url,
            'author': post.get('author', 'Unknown'),
            'text': post.get('text', ''),
            'classification': result['classification'],
            'confidence': result['confidence'],
            'verdict': result.get('verdict', 'Unsure'),
            'status': result.get('status', 'unsure'),
            'model': model_used,
            'reasoning': result.get('reasoning') or result.get('explanation') or result.get('raw'),
            'rag_used': rag_used,
            'likes': post.get('metrics', {}).get('like_count'),
            'retweets': post.get('metrics', {}).get('retweet_count'),
            'reply_count': post.get('metrics', {}).get('reply_count'),
            'quote_count': post.get('metrics', {}).get('quote_count'),
            'bookmark_count': post.get('metrics', {}).get('bookmark_count'),
            'impression_count': post.get('metrics', {}).get('impression_count'),
        }

        if rag_used:
            response['rag_document'] = 'docs/rag-failsafe.md'
            response['highlight_word'] = 'rag'
        
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
    # This health endpoint reports which detectors are available so the frontend can show the backend status.
    models_loaded = []
    if twhin_detector:
        models_loaded.append("TwHIN-BERT")
    if qwen_detector:
        models_loaded.append("Qwen3-VL")
    
    return {
        'status': 'ok',
        'models': models_loaded,
        'primary_model': models_loaded[0] if models_loaded else 'none'
    }

"""
Was adapted from examples at:

- https://fastapi.tiangolo.com/ (FastAPI - FastAPI, no date)
- https://docs.x.com/x-api/introduction (Platform, no date)
"""