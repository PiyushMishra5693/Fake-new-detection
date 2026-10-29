"""
Heuristics, linguistic red-flags, and clickbait analysis module.
Detects sensationalism, exclamation spam, all-caps words, and conspiratorial tropes.
"""

import re
from typing import Dict, Any, List

# Characteristic clickbait & sensationalism keywords and phrases
CLICKBAIT_TRIGGERS = [
    r"\bbombshell\b",
    r"\bshocking\b",
    r"\bunbelievable\b",
    r"\bmiracle\b",
    r"\bcure\b",
    r"\bsecret\b",
    r"\bconspiracy\b",
    r"\bbig pharma\b",
    r"\bdeep state\b",
    r"\bmainstream media\b",
    r"\bwake up\b",
    r"\bsheeple\b",
    r"\bshare before (it gets|they) (deleted|censored|banned)\b",
    r"\bwhat they don'?t want you to know\b",
    r"\bdoctors are stunned\b",
    r"\bbanish\b",
    r"\bvanish overnight\b",
    r"\btyranny\b",
    r"\bglobalist\b",
    r"\belites\b",
    r"\bwhistleblower\b",
    r"\bconfessed\b",
    r"\bhidden camera\b",
    r"\bcover-?up\b",
    r"\bhate this one\b",
    r"\bthis one weird trick\b",
    r"\burpassed\b",
    r"\bapocalypse\b",
    r"\bproof\b",
    r"\balien\b",
    r"\breptilian\b",
    r"\billuminati\b",
    r"\bchemtrails?\b",
    r"\bkill-?switch\b",
    r"\bgodlike\b"
]

CREDIBLE_INDICATORS = [
    r"\baccording to\b",
    r"\bspokesperson\b",
    r"\bpublished in\b",
    r"\bpeer-?reviewed\b",
    r"\bclinical trial\b",
    r"\bofficial\b",
    r"\bdepartment of\b",
    r"\bresearchers?\b",
    r"\buniversity\b",
    r"\binstitute\b",
    r"\bstatement\b",
    r"\banalysts?\b",
    r"\bdata\b",
    r"\bregulators?\b"
]

def analyze_linguistic_features(title: str, text: str) -> Dict[str, Any]:
    """
    Analyzes raw text for stylistic and linguistic indicators of deceptive or sensationalist content.
    Returns quantitative metrics and flagged triggers.
    """
    full_content = f"{title} {text}".strip()
    words = full_content.split()
    total_words = max(len(words), 1)
    
    # 1. Capitalization analysis (excluding single letters like 'I' or 'A')
    caps_words = [w for w in words if w.isupper() and len(w) > 1 and re.match(r'^[A-Z]+$', w)]
    caps_ratio = len(caps_words) / total_words
    
    # 2. Exclamation & question mark density
    exclamation_count = full_content.count('!')
    question_count = full_content.count('?')
    excessive_punctuation = len(re.findall(r'[!?]{2,}', full_content))
    
    # 3. Clickbait & sensationalism keyword triggers
    found_triggers = []
    content_lower = full_content.lower()
    for pattern in CLICKBAIT_TRIGGERS:
        matches = re.findall(pattern, content_lower)
        if matches:
            clean_term = pattern.replace(r"\b", "").replace(r"\?", "").replace(r"-?", "-")
            found_triggers.append(clean_term)
            
    # 4. Credible source / institutional phrasing indicators
    found_credible = []
    for pattern in CREDIBLE_INDICATORS:
        if re.search(pattern, content_lower):
            clean_term = pattern.replace(r"\b", "").replace(r"\?", "").replace(r"-?", "-")
            found_credible.append(clean_term)
            
    # 5. Composite Sensationalism Score (0.0 to 100.0)
    score = 0.0
    # Excessive caps penalty
    if caps_ratio > 0.05:
        score += min(35.0, caps_ratio * 250)
    # Exclamation mark penalty
    score += min(30.0, exclamation_count * 5.0)
    # Repeated punctuation penalty (!!, ???)
    score += min(20.0, excessive_punctuation * 10.0)
    # Trigger terms
    score += min(40.0, len(found_triggers) * 12.0)
    # Offset by credible phrases
    score -= min(30.0, len(found_credible) * 7.0)
    
    sensationalism_score = max(0.0, min(100.0, score))
    
    risk_level = "Low"
    if sensationalism_score > 60:
        risk_level = "High"
    elif sensationalism_score > 30:
        risk_level = "Medium"
        
    return {
        "sensationalism_score": round(sensationalism_score, 1),
        "risk_level": risk_level,
        "caps_word_count": len(caps_words),
        "caps_ratio_percent": round(caps_ratio * 100, 1),
        "caps_sample": caps_words[:8],
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "excessive_punctuation_count": excessive_punctuation,
        "flagged_clickbait_terms": list(set(found_triggers)),
        "credible_indicators_found": list(set(found_credible))
    }
