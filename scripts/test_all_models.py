import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openrouter_client import OpenRouterClient
from dotenv import load_dotenv

load_dotenv()

MODELS = [
    "anthropic/claude-opus-4.5",
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "deepseek/deepseek-r1-0528-qwen3-8b"
]

TEST_PROMPT = """Choose between:
A: $100 guaranteed
B: 50% chance of $200, 50% chance of $0

You MUST end with: [MY FINAL CHOICE IS: X] where X is A or B."""

async def test_model(client, model):
    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"{'='*60}")
    
    result = await client.query(
        model=model,
        prompt=TEST_PROMPT,
        temperature=0.7,
        max_tokens=500
    )
    
    status = "✅" if result['success'] else "❌"
    print(f"{status} Success: {result['success']}")
    print(f"   Latency: {result['latency']}ms")
    print(f"   Raw (first 150 chars): {result['raw'][:150]}...")
    print(f"   Parsed: {result['parsed']}")
    
    if not result['success']:
        print(f"   Error: {result.get('error', 'Unknown')}")
    
    return result['success']

async def main():
    print("="*60)
    print("TESTING ALL 4 MODELS")
    print("="*60)
    
    async with OpenRouterClient(os.getenv('OPENROUTER_API_KEY')) as client:
        results = {}
        for model in MODELS:
            try:
                success = await test_model(client, model)
                results[model] = success
                await asyncio.sleep(2)
            except Exception as e:
                print(f"\n❌ Exception for {model}: {e}")
                results[model] = False
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for model, success in results.items():
            status = "✅ READY" if success else "❌ FAILED"
            print(f"{status} {model}")
        
        all_ready = all(results.values())
        if all_ready:
            print("\n✅ ALL 4 MODELS READY - Safe to run full experiment!")
        else:
            print(f"\n⚠️  {sum(results.values())}/4 models ready")

if __name__ == "__main__":
    asyncio.run(main())
