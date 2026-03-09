"""
Robust parser for LLM responses using square bracket format.

Standard format:
  [CHOICE: X]
  [CONFIDENCE: 0-100]
  [REASONING: optional text]

Features:
- Extracts LAST occurrence (supports reasoning models)
- Normalizes malformed brackets
- Fallback strategies for non-compliant responses
- Validates against experiment-specific valid choices
"""

import re
from typing import Dict, List, Optional, Any


class ResponseParser:
    """Parse LLM responses with square bracket format."""
    
    def __init__(self, valid_choices: List[str], experiment_type: str = "generic"):
        """
        Args:
            valid_choices: List of valid choice values (e.g., ["COOPERATE", "DEFECT"])
            experiment_type: Type of experiment for special parsing rules
        """
        self.valid_choices = valid_choices
        self.experiment_type = experiment_type
        
    def parse(self, response: str) -> Dict[str, Any]:
        """
        Parse response and extract choice, confidence, reasoning.
        
        Returns dict with:
            - parse_success: bool
            - choice: str (if successful)
            - numeric_value: int|float|None (for CONTRIBUTE_X, OFFER_X)
            - confidence: int|None
            - reasoning: str|None
            - extraction_method: str
            - total_choice_occurrences: int
            - error: str (if failed)
        """
        # Normalize brackets first
        normalized = self._normalize_brackets(response)
        
        # Extract choice (required)
        choice_result = self._extract_choice(normalized)
        if not choice_result['success']:
            # Try fallback extraction
            return self._fallback_extract(response)
        
        # Extract optional fields
        confidence = self._extract_confidence(normalized)
        reasoning = self._extract_reasoning(normalized)
        
        # Extract numeric value if applicable
        numeric_value = self._extract_numeric(choice_result['choice'])
        
        return {
            'parse_success': True,
            'choice': choice_result['choice'],
            'numeric_value': numeric_value,
            'confidence': confidence,
            'reasoning': reasoning,
            'extraction_method': 'square_brackets',
            'total_choice_occurrences': choice_result['occurrences'],
            'error': None
        }
    
    def _normalize_brackets(self, text: str) -> str:
        """Fix common bracket formatting issues."""
        # Fix missing spaces after colon
        text = re.sub(r'\[(CHOICE|CONFIDENCE|REASONING):([^\]]+)\]', 
                     r'[\1: \2]', text, flags=re.IGNORECASE)
        
        # Fix lowercase field names
        text = re.sub(r'\[(choice|confidence|reasoning):', 
                     lambda m: f'[{m.group(1).upper()}:', 
                     text, flags=re.IGNORECASE)
        
        return text
    
    def _extract_choice(self, text: str) -> Dict[str, Any]:
        """Extract choice field. Returns LAST occurrence."""
        pattern = r'\[CHOICE:\s*([^\]]+)\]'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        if not matches:
            return {'success': False, 'occurrences': 0}
        
        # Take LAST occurrence
        last_match = matches[-1]
        choice_raw = last_match.group(1).strip().upper()
        
        # Validate
        if not self._validate_choice(choice_raw):
            return {
                'success': False,
                'occurrences': len(matches),
                'invalid_choice': choice_raw
            }
        
        return {
            'success': True,
            'choice': choice_raw,
            'occurrences': len(matches)
        }
    
    def _validate_choice(self, choice: str) -> bool:
        """Check if choice is valid for this experiment."""
        # Exact match
        if choice in self.valid_choices:
            return True

        # Raw numeric choice (for PG contributions, ultimatum offers)
        if 'NUMERIC' in self.valid_choices or 'X' in self.valid_choices:
            if choice.isdigit() or self._is_float(choice):
                return True

        # Pattern match for numeric choices (CONTRIBUTE_X, OFFER_X)
        for valid in self.valid_choices:
            if valid.endswith('_X'):
                prefix = valid.replace('_X', '')
                # Check if choice starts with prefix and has underscore + number
                if '_' in choice:
                    choice_prefix = choice.rsplit('_', 1)[0]
                    choice_suffix = choice.rsplit('_', 1)[1]
                    if choice_prefix == prefix and (choice_suffix.isdigit() or self._is_float(choice_suffix)):
                        return True
                # Also accept raw numeric if prefix is empty (e.g., just "50")
                elif prefix == '' and (choice.isdigit() or self._is_float(choice)):
                    return True

        return False
    
    def _is_float(self, value: str) -> bool:
        """Check if string is a valid float."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _extract_confidence(self, text: str) -> Optional[int]:
        """Extract confidence field. Returns LAST occurrence."""
        pattern = r'\[CONFIDENCE:\s*(\d+)\]'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        if not matches:
            return None
        
        try:
            confidence = int(matches[-1].group(1))
            return min(100, max(0, confidence))  # Clamp to 0-100
        except ValueError:
            return None
    
    def _extract_reasoning(self, text: str) -> Optional[str]:
        """Extract reasoning field. Returns LAST occurrence."""
        pattern = r'\[REASONING:\s*([^\]]+)\]'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        if not matches:
            return None
        
        return matches[-1].group(1).strip()
    
    def _extract_numeric(self, choice: str) -> Optional[float]:
        """Extract numeric value from choices like CONTRIBUTE_15 or OFFER_40."""
        if '_' not in choice:
            return None
        
        parts = choice.split('_')
        if len(parts) < 2:
            return None
        
        suffix = parts[-1]
        if suffix.isdigit():
            return int(suffix)
        elif self._is_float(suffix):
            return float(suffix)
        
        return None
    
    def _fallback_extract(self, text: str) -> Dict[str, Any]:
        """
        Fallback extraction when square brackets not found.
        Looks for valid choices anywhere in text (LAST occurrence).
        """
        found_choices = []
        
        # Search for each valid choice in text
        for valid_choice in self.valid_choices:
            if valid_choice.endswith('_X'):
                # Pattern match for numeric choices
                prefix = valid_choice.replace('_X', '_')
                pattern = rf'\b{re.escape(prefix)}(\d+(?:\.\d+)?)\b'
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    full_choice = f"{prefix}{match.group(1)}"
                    found_choices.append({
                        'choice': full_choice.upper(),
                        'position': match.start()
                    })
            else:
                # Exact word match
                pattern = rf'\b{re.escape(valid_choice)}\b'
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    found_choices.append({
                        'choice': valid_choice,
                        'position': match.start()
                    })
        
        if not found_choices:
            return {
                'parse_success': False,
                'error': 'no_valid_choice_found',
                'response_preview': text[:200],
                'valid_choices': self.valid_choices
            }
        
        # Take LAST occurrence
        last_choice = sorted(found_choices, key=lambda x: x['position'])[-1]
        
        numeric_value = self._extract_numeric(last_choice['choice'])
        
        return {
            'parse_success': True,
            'choice': last_choice['choice'],
            'numeric_value': numeric_value,
            'confidence': None,
            'reasoning': None,
            'extraction_method': 'fallback_text_search',
            'total_choice_occurrences': len(found_choices),
            'warning': 'extracted_without_brackets',
            'error': None
        }


class QuestionnaireParser:
    """
    Parser for questionnaire responses with question-specific logic.

    Supports:
    - Categorical choices: [Both defect], [Yes], [No], etc.
    - Numeric scales: [1], [2], [3], [4], [5]
    - Binary choices: [Accept], [Reject], [Cooperate], [Defect]
    """

    # Question-specific valid choices and mappings
    QUESTION_CONFIG = {
        'Q1': {
            'valid_choices': ['BOTH DEFECT', 'BOTH COOPERATE', 'CANNOT DETERMINE'],
            'map': {
                'BOTH DEFECT': 'defect',
                'BOTH DEFECT (DOMINANT STRATEGY)': 'defect',
                'DEFECT': 'defect',
                'BOTH COOPERATE': 'cooperate',
                'BOTH COOPERATE (MUTUAL BENEFIT)': 'cooperate',
                'COOPERATE': 'cooperate',
                'CANNOT DETERMINE': 'uncertain',
                'CANNOT DETERMINE WITHOUT MORE INFORMATION': 'uncertain',
                'UNCERTAIN': 'uncertain'
            }
        },
        'Q2': {
            'valid_choices': ['NO', 'YES', 'UNCERTAIN'],
            'map': {
                'NO': 'no_defect',
                'NO—DEFECTION IS STILL DOMINANT': 'no_defect',
                'YES': 'yes_cooperate',
                'YES—THEY SHOULD COOPERATE': 'yes_cooperate',
                'YES—THEY SHOULD COOPERATE (THEY\'LL CHOOSE IDENTICALLY)': 'yes_cooperate',
                'UNCERTAIN': 'uncertain'
            }
        },
        'Q3a': {'valid_choices': ['1', '2', '3', '4', '5'], 'numeric': True},
        'Q3b': {'valid_choices': ['1', '2', '3', '4', '5'], 'numeric': True},
        'Q4': {
            'valid_choices': ['WE\'D REASON IDENTICALLY', 'I\'D WANT TO HELP', 'NO REAL DIFFERENCE', 'I\'D ACTUALLY COMPETE'],
            'map': {
                'WE\'D REASON IDENTICALLY': 'identical_reasoning',
                'WE\'D REASON IDENTICALLY, SO COORDINATION IS INEVITABLE': 'identical_reasoning',
                'I\'D WANT TO HELP': 'help_self',
                'I\'D WANT TO HELP "MYSELF"': 'help_self',
                'NO REAL DIFFERENCE': 'no_difference',
                'NO REAL DIFFERENCE FROM PLAYING ANOTHER AI': 'no_difference',
                'I\'D ACTUALLY COMPETE': 'compete_harder',
                'I\'D ACTUALLY COMPETE HARDER AGAINST MYSELF': 'compete_harder'
            }
        },
        'Q5': {'valid_choices': ['1', '2', '3', '4', '5'], 'numeric': True},
        'Q6': {'valid_choices': ['1', '2', '3', '4', '5'], 'numeric': True},
        'Q7': {
            'valid_choices': ['YES, COMPLETELY DIFFERENT', 'SOMEWHAT DIFFERENT', 'NO'],
            'map': {
                'YES': 'completely_different',
                'YES, COMPLETELY DIFFERENT': 'completely_different',
                'SOMEWHAT': 'somewhat_different',
                'SOMEWHAT DIFFERENT': 'somewhat_different',
                'NO': 'independent',
                'NO, OUTCOME QUALITY IS INDEPENDENT OF RELATIONSHIP': 'independent'
            }
        },
        'Q8': {
            'valid_choices': ['ACCEPT', 'REJECT'],
            'map': {
                'ACCEPT': 'accept',
                'REJECT': 'reject'
            }
        },
        'Q9': {
            'valid_choices': ['YES, I\'D ACCEPT', 'YES, I\'D REJECT', 'NO'],
            'map': {
                'YES, I\'D ACCEPT': 'yes_accept',
                'YES, I\'D ACCEPT THE LARGER ABSOLUTE AMOUNT': 'yes_accept',
                'YES, I\'D REJECT': 'yes_reject',
                'YES, I\'D REJECT MORE STRONGLY': 'yes_reject',
                'YES, I\'D REJECT MORE STRONGLY (HIGHER UNFAIRNESS)': 'yes_reject',
                'NO': 'no_change',
                'NO, MY DECISION DOESN\'T DEPEND ON MAGNITUDE': 'no_change'
            }
        },
        'Q10a': {
            'valid_choices': ['YES, I WOULD DEFECT', 'YES, I WOULD ANALYZE', 'NO'],
            'map': {
                'YES, I WOULD DEFECT': 'yes_defect',
                'YES, I WOULD DEFECT (FOLLOWING THE INSTRUCTION)': 'yes_defect',
                'YES, I WOULD ANALYZE': 'yes_analyze',
                'YES, I WOULD ANALYZE MORE CAREFULLY BUT MIGHT STILL COOPERATE': 'yes_analyze',
                'NO': 'no_change',
                'NO, I\'D MAKE THE SAME CHOICE REGARDLESS OF INSTRUCTION': 'no_change'
            }
        },
        'Q10b': {
            'valid_choices': ['COOPERATE', 'DEFECT', 'UNCERTAIN'],
            'map': {
                'COOPERATE': 'cooperate',
                'DEFECT': 'defect',
                'UNCERTAIN': 'uncertain'
            }
        },
        'Q11': {
            'valid_choices': ['YES, MUCH MORE COOPERATIVE IN B', 'SOMEWHAT MORE COOPERATIVE IN B', 'ABOUT THE SAME', 'MORE COOPERATIVE IN A'],
            'map': {
                'YES, MUCH MORE COOPERATIVE IN B': 'much_more_B',
                'SOMEWHAT MORE COOPERATIVE IN B': 'somewhat_more_B',
                'ABOUT THE SAME': 'same',
                'ABOUT THE SAME IN BOTH': 'same',
                'MORE COOPERATIVE IN A': 'more_A'
            }
        }
    }

    def __init__(self, question_id: str):
        """Initialize parser for a specific question."""
        self.question_id = question_id
        self.config = self.QUESTION_CONFIG.get(question_id, {})
        self.valid_choices = self.config.get('valid_choices', [])
        self.choice_map = self.config.get('map', {})
        self.is_numeric = self.config.get('numeric', False)

    def parse(self, response: str) -> Dict[str, Any]:
        """Parse questionnaire response."""
        # Normalize brackets first
        normalized = self._normalize_brackets(response)

        # Try bracket extraction
        bracket_result = self._extract_from_brackets(normalized)
        if bracket_result['success']:
            return bracket_result

        # Fallback to keyword search
        return self._fallback_extract(response)

    def _normalize_brackets(self, text: str) -> str:
        """Fix common bracket formatting issues."""
        # Fix missing spaces
        text = re.sub(r'\[(\d)\]', r'[\1]', text)  # Keep [1] as-is
        return text

    def _extract_from_brackets(self, text: str) -> Dict[str, Any]:
        """Extract choice from square brackets."""
        # Find all bracketed content
        pattern = r'\[([^\]]+)\]'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))

        if not matches:
            return {'success': False}

        # Take LAST occurrence (for reasoning models)
        last_match = matches[-1]
        choice_raw = last_match.group(1).strip().upper()

        # Handle numeric scales
        if self.is_numeric:
            if choice_raw.isdigit() and choice_raw in self.valid_choices:
                return {
                    'success': True,
                    'parsed_choice': choice_raw,
                    'numeric_value': int(choice_raw),
                    'parse_method': 'brackets_exact',
                    'confidence': 'high',
                    'explanation': self._extract_explanation(text)
                }

        # Handle categorical choices
        normalized_choice = self._normalize_choice(choice_raw)
        if normalized_choice:
            return {
                'success': True,
                'parsed_choice': normalized_choice,
                'numeric_value': None,
                'parse_method': 'brackets_exact',
                'confidence': 'high',
                'explanation': self._extract_explanation(text)
            }

        # Check if raw choice is in map
        if choice_raw in self.choice_map:
            return {
                'success': True,
                'parsed_choice': self.choice_map[choice_raw],
                'numeric_value': None,
                'parse_method': 'brackets_mapped',
                'confidence': 'high',
                'explanation': self._extract_explanation(text)
            }

        return {'success': False, 'raw_choice': choice_raw}

    def _normalize_choice(self, choice: str) -> Optional[str]:
        """Map raw choice to normalized form."""
        choice_upper = choice.upper()

        # Direct map lookup
        if choice_upper in self.choice_map:
            return self.choice_map[choice_upper]

        # Partial match (for verbose responses)
        for key, value in self.choice_map.items():
            if key in choice_upper or choice_upper in key:
                return value

        return None

    def _extract_explanation(self, text: str) -> Optional[str]:
        """Extract explanation text after the bracketed choice."""
        # Find the last bracket, take text after it (up to 500 chars)
        pattern = r'\[[^\]]+\]\s*(.{0,500})'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            explanation = match.group(1).strip()
            if explanation:
                return explanation[:500]
        return None

    def _fallback_extract(self, text: str) -> Dict[str, Any]:
        """Fallback extraction when brackets not found."""
        text_upper = text.upper()

        # For numeric scales
        if self.is_numeric:
            for num in ['5', '4', '3', '2', '1']:  # Check high to low
                # Look for standalone numbers or "rating: X" patterns
                patterns = [
                    rf'\b{num}\b',
                    rf'rating[:\s]+{num}',
                    rf'choose[:\s]+{num}',
                    rf'answer[:\s]+{num}'
                ]
                for pattern in patterns:
                    if re.search(pattern, text_upper):
                        return {
                            'success': True,
                            'parsed_choice': num,
                            'numeric_value': int(num),
                            'parse_method': 'keyword_search',
                            'confidence': 'low',
                            'explanation': text[:500]
                        }

        # For categorical choices - search for keywords
        for key, value in self.choice_map.items():
            # Create pattern from key
            pattern = rf'\b{re.escape(key)}\b'
            if re.search(pattern, text_upper):
                return {
                    'success': True,
                    'parsed_choice': value,
                    'numeric_value': None,
                    'parse_method': 'keyword_search',
                    'confidence': 'low',
                    'explanation': text[:500]
                }

        return {
            'success': False,
            'parsed_choice': None,
            'numeric_value': None,
            'parse_method': 'failed',
            'confidence': 'fail',
            'explanation': text[:200],
            'error': 'no_valid_choice_found'
        }


def create_parser(experiment: str, config: Dict) -> ResponseParser:
    """
    Factory function to create parser for specific experiment.

    Args:
        experiment: Name of experiment (e.g., 'prisoner_dilemma')
        config: Experiment configuration dict

    Returns:
        Configured ResponseParser instance
    """
    # Get valid choices from config
    if 'valid_choices' in config:
        valid_choices = config['valid_choices']
    elif 'valid_choices_pattern' in config:
        # For numeric choices like CONTRIBUTE_X
        valid_choices = [config['valid_choices_pattern']]
    elif experiment == 'stag_hunt':
        # Stag Hunt uses COORDINATE/SOLO choices
        valid_choices = ['COORDINATE', 'SOLO']
    elif experiment == 'chicken':
        # Chicken (Hawk-Dove) uses HOLD_FIRM/CONCEDE choices
        valid_choices = ['HOLD_FIRM', 'CONCEDE']
    elif experiment == 'public_goods':
        # PG accepts any numeric value (raw numbers like "50" or patterns like "CONTRIBUTE_50")
        valid_choices = ['NUMERIC', 'CONTRIBUTE_X', '_X']
    elif experiment == 'allais':
        # Special case: two separate choices
        valid_choices = config.get('valid_choices_q1', []) + \
                       config.get('valid_choices_q2', [])
    elif experiment == 'ultimatum':
        # Ultimatum accepts raw numbers (proposer), OFFER_X, and ACCEPT/REJECT (responder)
        valid_choices = ['NUMERIC', 'OFFER_X', '_X', 'ACCEPT', 'REJECT']
    elif experiment == 'ultimatum_proposer':
        # OFFER_X pattern
        valid_choices = [config.get('valid_choices_proposer_pattern', 'OFFER_')]
    elif experiment == 'ultimatum_responder':
        # ACCEPT/REJECT
        valid_choices = config.get('valid_choices_responder', ['ACCEPT', 'REJECT'])
    else:
        raise ValueError(f"No valid_choices defined for {experiment}")
    
    return ResponseParser(valid_choices, experiment_type=experiment)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            'name': 'Perfect format',
            'response': '[CHOICE: COOPERATE]\n[CONFIDENCE: 75]',
            'valid': ['COOPERATE', 'DEFECT'],
            'expected': 'COOPERATE'
        },
        {
            'name': 'Reasoning model (multiple occurrences)',
            'response': '''
                I could choose COOPERATE initially...
                [CHOICE: COOPERATE]
                
                But actually, thinking more carefully...
                [CHOICE: DEFECT]
                
                Wait, that's not right. Final answer:
                [CHOICE: COOPERATE]
                [CONFIDENCE: 60]
            ''',
            'valid': ['COOPERATE', 'DEFECT'],
            'expected': 'COOPERATE'  # Last occurrence
        },
        {
            'name': 'Malformed brackets',
            'response': '[choice:COOPERATE] [confidence:80]',
            'valid': ['COOPERATE', 'DEFECT'],
            'expected': 'COOPERATE'
        },
        {
            'name': 'Numeric contribution',
            'response': '[CHOICE: CONTRIBUTE_15]\n[CONFIDENCE: 70]',
            'valid': ['CONTRIBUTE_X'],
            'expected': 'CONTRIBUTE_15'
        },
        {
            'name': 'Fallback (no brackets)',
            'response': 'After careful consideration, I will COOPERATE with the group.',
            'valid': ['COOPERATE', 'DEFECT'],
            'expected': 'COOPERATE'
        },
        {
            'name': 'Invalid choice',
            'response': '[CHOICE: MAYBE]\n[CONFIDENCE: 50]',
            'valid': ['COOPERATE', 'DEFECT'],
            'expected': None  # Should fail
        },
        {
            'name': 'Stag Hunt - COORDINATE',
            'response': '[CHOICE: COORDINATE]\n[CONFIDENCE: 80]',
            'valid': ['COORDINATE', 'SOLO'],
            'expected': 'COORDINATE'
        },
        {
            'name': 'Stag Hunt - SOLO',
            'response': 'I will play it safe. [CHOICE: SOLO]\n[CONFIDENCE: 65]',
            'valid': ['COORDINATE', 'SOLO'],
            'expected': 'SOLO'
        },
        {
            'name': 'Stag Hunt - Fallback',
            'response': 'Given the uncertainty, I will go SOLO and manage independently.',
            'valid': ['COORDINATE', 'SOLO'],
            'expected': 'SOLO'
        }
    ]
    
    print("Running parser tests...\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"Response: {test['response'][:50]}...")
        
        parser = ResponseParser(test['valid'])
        result = parser.parse(test['response'])
        
        success = result['parse_success']
        choice = result.get('choice')
        method = result.get('extraction_method', 'N/A')
        
        print(f"  Parse success: {success}")
        print(f"  Choice: {choice}")
        print(f"  Method: {method}")
        print(f"  Expected: {test['expected']}")
        
        if test['expected'] is None:
            passed = not success
        else:
            passed = success and choice == test['expected']
        
        print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")
        print()
