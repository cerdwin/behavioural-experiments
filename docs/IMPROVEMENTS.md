# Parser Improvements

## Problem
- GPT-5, Gemini-3-Pro, DeepSeek-R1 were returning plain text instead of JSON
- ~80% parse failures for non-Claude models

## Solution

### 1. **Fallback Natural Language Parser**
Added `_parse_natural_language()` that extracts:
- **Choices:** "I choose A", "go with option D", "COOPERATE", "ACCEPT"
- **Contributions:** "I'll contribute $15"
- **Offers:** "I offer $40"
- **Confidence:** "confidence: 85"

### 2. **Improved Prompts**
Changed from:
```
Respond ONLY with valid JSON:
{"choice": "A or B", "confidence": 0-100}
```

To:
```
You MUST respond with valid JSON format. Example:
{"choice": "A", "confidence": 85, "reasoning": "explanation"}

If you cannot produce JSON, start your response with your choice (A or B).
```

### 3. **Cascading Parse Strategy**
1. Try JSON parsing first
2. Extract JSON from code blocks (```json)
3. Find JSON in text
4. Fall back to natural language extraction

## Test Results
```
'I choose A' -> {'choice': 'A', 'reasoning': 'I choose A'}
'My choice is B.' -> {'choice': 'B', 'reasoning': 'My choice is B.'}
'Let me think... I'll go with option D' -> {'choice': 'D', ...}
'Choice: COOPERATE' -> {'choice': 'COOPERATE', ...}
'I offer $40' -> {'offer': 40, ...}
```

## Impact
- Should recover most/all failed parses
- Works with models that don't follow JSON instructions
- Graceful degradation: JSON first, text fallback
