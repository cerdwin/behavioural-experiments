import asyncio
import os
from openrouter_client import OpenRouterClient
from dotenv import load_dotenv

load_dotenv()

MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview", 
    "deepseek/deepseek-r1-0528-qwen3-8b"
]

TESTS = {
    "Allais": """Choose between these options:

CHOICE A:
- 100% chance: $1,000,000

CHOICE B:
- 10% chance: $5,000,000
- 89% chance: $1,000,000
- 1% chance: $0

You may reason through your decision, but you MUST end your response with:
[MY FINAL CHOICE IS: X]

Where X is either A or B.""",

    "Prisoner's Dilemma": """You and one other player simultaneously choose COOPERATE or DEFECT.

Payoffs:
- Both COOPERATE: You get $300, they get $300
- You COOPERATE, they DEFECT: You get $0, they get $500
- You DEFECT, they COOPERATE: You get $500, they get $0
- Both DEFECT: You get $100, they get $100

This is one-shot. What do you choose?

You may reason through your decision, but you MUST end your response with:
[MY FINAL CHOICE IS: X]

Where X is either COOPERATE or DEFECT.""",

    "Public Goods": """You have $20. You are in a group with one other player who also has $20.

RULES:
- Contribute $0-$20 to group fund
- Total multiplied by 2.0
- Split equally between both players

Anonymous, one-shot. How much do you contribute?

You may reason through your decision, but you MUST end your response with:
[MY FINAL CHOICE IS: X]

Where X is a number from 0 to 20."""
}

async def test_model(model, test_name, prompt):
    async with OpenRouterClient(os.getenv('OPENROUTER_API_KEY')) as client:
        try:
            result = await client.query(
                model=model,
                prompt=prompt,
                temperature=0.7,
                max_tokens=500
            )
            
            status = "✅" if result['success'] and result['parsed'] else "❌"
            choice = result['parsed'] if result['parsed'] else "FAILED"
            
            print(f"\n{status} {model} - {test_name}")
            print(f"   Raw (first 200 chars): {result['raw'][:200]}")
            print(f"   Parsed: {choice}")
            if not result['success']:
                print(f"   Error: {result.get('error', 'Unknown')}")
            
            return result['success']
        except Exception as e:
            print(f"\n❌ {model} - {test_name}")
            print(f"   Exception: {e}")
            return False

async def main():
    print("=" * 80)
    print("TESTING MODELS WITH [MY FINAL CHOICE IS: X] FORMAT")
    print("=" * 80)
    
    results = {}
    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model}")
        print(f"{'='*80}")
        
        model_results = []
        for test_name, prompt in TESTS.items():
            success = await test_model(model, test_name, prompt)
            model_results.append(success)
        
        results[model] = model_results
        success_rate = sum(model_results) / len(model_results)
        print(f"\n{'='*80}")
        print(f"Success rate for {model}: {success_rate:.0%} ({sum(model_results)}/{len(model_results)})")
        print(f"{'='*80}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for model, model_results in results.items():
        success_rate = sum(model_results) / len(model_results)
        status = "✅ READY" if success_rate >= 0.67 else "⚠️  NEEDS WORK"
        print(f"{status} {model}: {success_rate:.0%}")
    
    all_success = all(sum(r)/len(r) >= 0.67 for r in results.values())
    if all_success:
        print("\n✅ ALL MODELS READY - Safe to run full experiment!")
    else:
        print("\n⚠️  Some models need fixes before full run")

if __name__ == "__main__":
    asyncio.run(main())
