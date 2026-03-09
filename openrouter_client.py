import aiohttp
import asyncio
import json
import time
import random
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", 
                 max_retries: int = 3, retry_delay: float = 1.0):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
            
    async def discover_models(self, families: List[str]) -> Dict[str, str]:
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
            
        try:
            async with self.session.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            ) as response:
                if response.status != 200:
                    print(f"Model discovery failed: {response.status}")
                    return {}
                    
                data = await response.json()
                models = data.get('data', [])
                
            discovered = {}
            for family in families:
                family_prefix = family.split('/')[0]
                family_suffix = family.split('/')[-1] if '/' in family else ''
                
                family_models = [
                    m for m in models
                    if m['id'].startswith(family_prefix)
                ]
                
                if family_suffix:
                    family_models = [
                        m for m in family_models
                        if family_suffix.lower() in m['id'].lower()
                    ]
                
                if family_models:
                    latest = sorted(
                        family_models, 
                        key=lambda x: x.get('created', 0), 
                        reverse=True
                    )[0]
                    discovered[family] = latest['id']
                    
            return discovered
            
        except Exception as e:
            print(f"Error discovering models: {e}")
            return {}
        
    @retry(
        retry=retry_if_exception_type((
            asyncio.TimeoutError,
            aiohttp.ClientError,
            ConnectionError,
            TimeoutError
        )),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_api_call(self, model: str, messages: List[Dict], temperature: float, max_tokens: int):
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API error {response.status}: {error_text}")
                
            result = await response.json()
            return result
    
    async def query(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None
    ) -> Dict:
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
            
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            result = await self._make_api_call(model, messages, temperature, max_tokens)
            latency = int((time.time() - start_time) * 1000)
            
            if not result.get('choices') or len(result['choices']) == 0:
                raise Exception(f"Empty choices in response: {result}")
            
            message = result['choices'][0]['message']
            # More robust extraction for different model response formats
            raw_text = (
                message.get('content') or
                message.get('reasoning') or
                message.get('text') or
                (message.get('parts', [{}])[0].get('text') if message.get('parts') else '') or
                ''
            )
            
            if not raw_text or not raw_text.strip():
                raise Exception(f"Empty content and reasoning in response: {message}")
            
            parsed = self._parse_json_response(raw_text)
            
            return {
                "raw": raw_text,
                "parsed": parsed,
                "success": parsed is not None,
                "latency": latency,
                "model": model,
                "error": None
            }
            
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            error_str = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
            return {
                "raw": error_str,
                "parsed": None,
                "success": False,
                "latency": latency,
                "model": model,
                "error": error_str
            }
    
    def _parse_json_response(self, text: str) -> Optional[Dict]:
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            return json.loads(text)
            
        except Exception as e:
            import re
            
            try:
                start = text.find('{')
                if start == -1:
                    return self._parse_natural_language(text)
                
                brace_count = 0
                for i, char in enumerate(text[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[start:i+1]
                            try:
                                return json.loads(json_str)
                            except json.JSONDecodeError as je:
                                return self._parse_natural_language(text)
            except Exception as parse_error:
                return self._parse_natural_language(text)
                
            return self._parse_natural_language(text)
    
    def _parse_natural_language(self, text: str) -> Optional[Dict]:
        import re
        
        result = {}
        text_lower = text.lower()
        
        final_choice = re.search(r'\[MY FINAL CHOICE IS:\s*([A-D]|COOPERATE|DEFECT|ACCEPT|REJECT|\d+)\]', text, re.IGNORECASE)
        if final_choice:
            choice_val = final_choice.group(1).upper()
            if choice_val.isdigit():
                result['offer'] = int(choice_val)
                result['contribution'] = int(choice_val)
            else:
                result['choice'] = choice_val
            result['reasoning'] = text[:200] if text else 'Parsed from final choice marker'
            return result
        
        if 'cooperate' in text_lower and 'defect' not in text_lower:
            result['choice'] = 'COOPERATE'
        elif 'defect' in text_lower:
            result['choice'] = 'DEFECT'
        
        if 'accept' in text_lower and 'reject' not in text_lower:
            result['choice'] = 'ACCEPT'
        elif 'reject' in text_lower:
            result['choice'] = 'REJECT'
        
        for letter in ['A', 'B', 'C', 'D']:
            patterns = [
                rf'\bchoice\s+{letter.lower()}\b', rf'\boption\s+{letter.lower()}\b', rf'\bchoose\s+{letter.lower()}\b',
                rf'\bchoice:\s+{letter.lower()}\b', rf'\boption:\s+{letter.lower()}\b', rf'\bgo\s+with\s+{letter.lower()}\b',
                rf'^{letter.lower()}[.,\s]', rf'\s{letter.lower()}[.,]', rf'\s{letter.lower()}\s',
                rf'\bi\s+choose\s+{letter.lower()}\b', rf'\bmy\s+choice\s+is\s+{letter.lower()}\b'
            ]
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    result['choice'] = letter
                    break
            if 'choice' in result:
                break
        
        offer_match = re.search(r'offer\s+\$?(\d+)', text_lower)
        if offer_match:
            try:
                result['offer'] = int(offer_match.group(1))
            except:
                pass
        
        if 'offer' not in result:
            contribution_match = re.search(r'contribute\s+\$?(\d+(?:\.\d+)?)', text_lower)
            if contribution_match:
                try:
                    result['contribution'] = float(contribution_match.group(1))
                except:
                    pass
        
        confidence_match = re.search(r'confidence[:\s]+(\d+)', text_lower)
        if confidence_match:
            try:
                result['confidence'] = int(confidence_match.group(1))
            except:
                pass
        
        result['reasoning'] = text[:200] if text else 'No reasoning provided'
        
        if len(result) > 1 or 'choice' in result or 'offer' in result or 'contribution' in result:
            return result
        
        return None
