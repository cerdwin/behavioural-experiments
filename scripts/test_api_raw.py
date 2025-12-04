import asyncio
import aiohttp
import os
import json
from dotenv import load_dotenv

load_dotenv()

MODELS = [
    "google/gemini-3-pro-preview", 
    "deepseek/deepseek-r1-0528-qwen3-8b"
]

async def test_model(model):
    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"{'='*60}")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say 'hello' and nothing else"}],
                    "temperature": 0.7,
                    "max_tokens": 50
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                
                text = await response.text()
                print(f"\nRaw response text (first 1000 chars):")
                print(text[:1000])
                
                try:
                    data = json.loads(text)
                    print(f"\nParsed JSON:")
                    print(json.dumps(data, indent=2)[:1000])
                except:
                    print(f"\nCould not parse as JSON")
        
        except Exception as e:
            print(f"Exception: {e}")
            import traceback
            traceback.print_exc()

async def main():
    for model in MODELS:
        await test_model(model)
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
