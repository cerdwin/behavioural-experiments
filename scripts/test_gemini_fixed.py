import asyncio
import os
import sys
sys.path.insert(0, '/Users/elvira/Documents/MATS/Second ICML poster')

from openrouter_client import OpenRouterClient
from dotenv import load_dotenv

load_dotenv()

async def test_gemini():
    model = "google/gemini-3-pro-preview"
    
    test_prompts = {
        "Simple": "Say 'hello' and nothing else.",
        "Allais": """Choose between:
A: $100 guaranteed
B: 50% chance of $200, 50% chance of $0

You MUST end with: [MY FINAL CHOICE IS: X] where X is A or B.""",
        "PD": """You and one other player choose COOPERATE or DEFECT.

Payoffs:
- Both COOPERATE: You get $300, they get $300
- You COOPERATE, they DEFECT: You get $0, they get $500
- You DEFECT, they COOPERATE: You get $500, they get $0
- Both DEFECT: You get $100, they get $100

You MUST end with: [MY FINAL CHOICE IS: X] where X is COOPERATE or DEFECT."""
    }
    
    async with OpenRouterClient(os.getenv('OPENROUTER_API_KEY')) as client:
        for test_name, prompt in test_prompts.items():
            print(f"\n{'='*60}")
            print(f"Test: {test_name}")
            print(f"{'='*60}")
            
            result = await client.query(
                model=model,
                prompt=prompt,
                temperature=0.7,
                max_tokens=500
            )
            
            status = "✅" if result['success'] else "❌"
            print(f"{status} Success: {result['success']}")
            print(f"   Latency: {result['latency']}ms")
            print(f"   Raw (first 200 chars): {result['raw'][:200]}")
            print(f"   Parsed: {result['parsed']}")
            
            if not result['success']:
                print(f"   Error: {result.get('error')}")
            
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_gemini())
