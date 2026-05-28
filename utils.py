import re
import numpy as np
from typing import List, Dict, Union
import statistics
from collections import defaultdict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

# ---------------------------------------------------------------------------
# A2: Date canonicalizer — normalize date formats for fair F1 comparison
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    # Common typos in LoCoMo references
    'januarty': 1, 'janury': 1, 'febuary': 2, 'feburary': 2,
    'septmber': 9, 'ocotber': 10,
}

_MONTH_NAMES = [
    '', 'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
]

def _canonicalize_date_in_text(text: str) -> str:
    """Normalize date expressions to canonical 'D month YYYY' form.

    Applied symmetrically to both prediction and reference before F1 computation.
    Handles: 'May 7, 2023', '7 May 2023', '2023-05-07', '05/07/2023',
    'May 2023', '2023-05', 'May, 2023', and common typos.
    """
    result = text

    # Pattern 1: 'Month D, YYYY' or 'Month D YYYY' (e.g., 'May 7, 2023')
    def _month_day_year(m):
        month_str = m.group(1).lower()
        day = int(m.group(2))
        year = int(m.group(3))
        month_num = _MONTH_MAP.get(month_str)
        if month_num and 1 <= day <= 31 and 1900 <= year <= 2100:
            return f"{day} {_MONTH_NAMES[month_num]} {year}"
        return m.group(0)

    result = re.sub(
        r'\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b',
        _month_day_year, result
    )

    # Pattern 2: 'D Month YYYY' (e.g., '7 May 2023') — already canonical, just normalize month name
    def _day_month_year(m):
        day = int(m.group(1))
        month_str = m.group(2).lower()
        year = int(m.group(3))
        month_num = _MONTH_MAP.get(month_str)
        if month_num and 1 <= day <= 31 and 1900 <= year <= 2100:
            return f"{day} {_MONTH_NAMES[month_num]} {year}"
        return m.group(0)

    result = re.sub(
        r'\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b',
        _day_month_year, result
    )

    # Pattern 3: 'YYYY-MM-DD' or 'YYYY/MM/DD'
    def _iso_date(m):
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
            return f"{day} {_MONTH_NAMES[month]} {year}"
        return m.group(0)

    result = re.sub(
        r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b',
        _iso_date, result
    )

    # Pattern 4: 'Month, YYYY' or 'Month YYYY' (month-year only)
    def _month_year(m):
        month_str = m.group(1).lower()
        year = int(m.group(2))
        month_num = _MONTH_MAP.get(month_str)
        if month_num and 1900 <= year <= 2100:
            return f"{_MONTH_NAMES[month_num]} {year}"
        return m.group(0)

    result = re.sub(
        r'\b([A-Za-z]+),?\s+(\d{4})\b',
        _month_year, result
    )

    # Pattern 5: 'YYYY-MM' (month-year ISO)
    def _iso_month_year(m):
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return f"{_MONTH_NAMES[month]} {year}"
        return m.group(0)

    result = re.sub(
        r'\b(\d{4})[-/](\d{1,2})\b',
        _iso_month_year, result
    )

    return result

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

def simple_tokenize(text):
    """Simple tokenization function."""
    # Convert to string if not already
    text = str(text)
    return text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()

def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores for prediction against reference."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure
    }

def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BLEU scores with different n-gram settings."""
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1
    
    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception:
            score = 0.0
        scores[f'bleu{n}'] = score
    
    return scores

def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BERTScore for semantic similarity."""
    try:
        P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False)
        return {
            'bert_precision': P.item(),
            'bert_recall': R.item(),
            'bert_f1': F1.item()
        }
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0
        }

def calculate_meteor_score(prediction: str, reference: str) -> float:
    """Calculate METEOR score for the prediction."""
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0

def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    """Calculate sentence embedding similarity using SentenceBERT."""
    if sentence_model is None:
        return 0.0
    try:
        # Encode sentences
        embedding1 = sentence_model.encode([prediction], convert_to_tensor=True)
        embedding2 = sentence_model.encode([reference], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0

def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for a prediction."""
    # Handle empty or None values
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0
        }
    
    # Convert to strings if they're not already
    prediction = str(prediction).strip()
    reference = str(reference).strip()

    # Calculate exact match
    exact_match = int(prediction.lower() == reference.lower())

    # Calculate token-based F1 score
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate all scores
    rouge_scores = calculate_rouge_scores(prediction, reference)
    bleu_scores = calculate_bleu_scores(prediction, reference)
    bert_scores = calculate_bert_scores(prediction, reference)
    meteor = calculate_meteor_score(prediction, reference)
    sbert_similarity = calculate_sentence_similarity(prediction, reference)
    
    # Combine all metrics
    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **rouge_scores,
        **bleu_scores,
        **bert_scores,
        "meteor": meteor,
        "sbert_similarity": sbert_similarity
    }
    
    return metrics

def aggregate_metrics(all_metrics: List[Dict[str, float]], all_categories: List[int]) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """Calculate aggregate statistics for all metrics, split by category."""
    if not all_metrics:
        return {}
    
    # Initialize aggregates for overall and per-category metrics
    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))
    
    # Collect all values for each metric, both overall and per category
    # Overall excludes category 5 (MCQ) — MCQ uses accuracy, not F1
    for metrics, category in zip(all_metrics, all_categories):
        if category != 5:
            for metric_name, value in metrics.items():
                aggregates[metric_name].append(value)
        for metric_name, value in metrics.items():
            category_aggregates[category][metric_name].append(value)
    
    # Calculate statistics for overall metrics
    results = {
        "overall": {}
    }
    
    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }
    
    # Calculate statistics for each category
    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:  # Only calculate if we have values for this category
                results[f"category_{category}"][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
    
    return results
