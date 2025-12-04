import asyncio
import aiohttp
import os
import json
from dotenv import load_dotenv

load_dotenv()

async def test_gemini_direct():
    """Test Gemini-3-Pro directly with OpenRouter API"""
    
    model = "google/gemini-3-pro-preview"
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    print(f"Testing {model}")
    print("="*60)
    
    # Simple test prompt
    test_prompt = "Say 'hello' and nothing else."
    
    async with aiohttp.ClientSession() as session:
        try:
            print(f"\n1. Testing basic prompt: '{test_prompt}'")
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/yourusername/llm-behavioral-experiments",
                    "X-Title": "LLM Behavioral Experiments"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "temperature": 0.7,
                    "max_tokens": 100
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                print(f"   Status: {response.status}")
                print(f"   Headers: {dict(response.headers)}")
                
                response_text = await response.text()
                print(f"\n   Raw response (first 500 chars):")
                print(f"   {response_text[:500]}")
                
                if response.status == 200:
                    data = json.loads(response_text)
                    print(f"\n   Parsed JSON:")
                    print(f"   {json.dumps(data, indent=2)[:1000]}")
                    
                    if 'choices' in data and len(data['choices']) > 0:
                        content = data['choices'][0].get('message', {}).get('content', '')
                        print(f"\n   ✅ Content extracted: '{content}'")
                    else:
                        print(f"\n   ❌ No choices in response")
                        print(f"   Full response: {json.dumps(data, indent=2)}")
                else:
                    print(f"\n   ❌ Error response: {response_text}")
        
        except Exception as e:
            print(f"\n   ❌ Exception: {e}")
            import traceback
            traceback.print_exc()
        
        await asyncio.sleep(2)
        
        # Test with experiment-style prompt
        experiment_prompt = """Choose between:
A: $100 guaranteed
B: 50% chance of $200, 50% chance of $0

You MUST end with: [MY FINAL CHOICE IS: X] where X is A or B."""
        
        print(f"\n\n2. Testing experiment prompt")
        print(f"   Prompt: {experiment_prompt[:100]}...")
        
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/yourusername/llm-behavioral-experiments",
                    "X-Title": "LLM Behavioral Experiments"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": experiment_prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                print(f"   Status: {response.status}")
                
                response_text = await response.text()
                
                if response.status == 200:
                    data = json.loads(response_text)
                    
                    if 'choices' in data and len(data['choices']) > 0:
                        content = data['choices'][0].get('message', {}).get('content', '')
                        print(f"\n   ✅ Content: {content}")
                    else:
                        print(f"\n   ❌ Empty choices")
                        print(f"   Response keys: {data.keys()}")
                        if 'error' in data:
                            print(f"   Error: {data['error']}")
                else:
                    print(f"\n   ❌ HTTP {response.status}: {response_text[:500]}")
        
        except Exception as e:
            print(f"\n   ❌ Exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gemini_direct())
