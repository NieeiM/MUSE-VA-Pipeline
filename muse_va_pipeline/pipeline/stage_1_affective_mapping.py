import json
import os
import sys

# # Add the va_to_caption package directory to Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# package_dir = os.path.dirname(current_dir)
# if package_dir not in sys.path:
#     sys.path.insert(0, package_dir)

sys.path.append('.')

# Try relative import first, fallback to absolute
try:
    from ..utils.va_emotion_mapper import va_to_emotions
    from ..utils.va_word_mapper import map_va_to_word
except ImportError:
    # When running as script
    from va_to_caption.utils.va_emotion_mapper import va_to_emotions
    from va_to_caption.utils.va_word_mapper import map_va_to_word


def load_prompt(stage_name, language='en'):
    """Loads a prompt template from the prompts directory.
    
    Args:
        stage_name: Name of the stage (e.g., 'stage_1')
        language: Language code ('en' for English, 'zh' for Chinese)
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__), '..', 'prompts',
        f"{stage_name}_{language}.txt"
    )
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def run_stage_1(
    valence: float,
    arousal: float,
    model_name: str = None,
    log_dir: str = None,
    language: str = 'en',
    k_emotion: int = 10,
    k_word: int = 30,
    random_seed: int = None
):
    """
    Executes Stage 1: Affective Mapping (Rule-based).

    Takes V-A coordinates and returns emotional palette using rule-based mapping.
    
    Args:
        valence: Valence value [1-9]
        arousal: Arousal value [1-9]
        model_name: Not used (kept for compatibility)
        log_dir: Directory for logging outputs
        language: Language code ('en' or 'zh') - not used in rule-based version
        k_emotion: Number of nearest samples for emotion mapping
        k_word: Number of nearest words for theme selection
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with format: {'theme': 'word', 'emotion': 'Emotion Label'}
    """
    print(f"--- Running Stage 1: Affective Mapping (Rule-based) ---")
    print(f"Input: Valence={valence}, Arousal={arousal}")

    # Get emotion using va_emotion_mapper (sample from top emotions)
    emotion = va_to_emotions(
        valence=valence,
        arousal=arousal,
        k=k_emotion,
        n=3,  # Consider top 3 emotions for sampling
        return_format='sample',
        random_seed=random_seed,
        verbose=False
    )

    # Get theme word using va_word_mapper
    theme = map_va_to_word(
        valence=valence, arousal=arousal, k=k_word, random_seed=random_seed
    )

    # Create response in the new format
    response_data = {'theme': theme, 'emotion': emotion}

    print(f"Generated: Theme='{theme}', Emotion='{emotion}'")

    # Log the output if log_dir is provided
    if log_dir:
        with open(os.path.join(log_dir, 'stage_1_output.json'), 'w') as f:
            json.dump(response_data, f, indent=4, ensure_ascii=False)

    # print("Successfully generated emotional mapping.")
    return response_data


if __name__ == "__main__":
    """Quick test of rule-based stage 1 mapping"""
    # Simple test to verify the function works
    result = run_stage_1(valence=7.5, arousal=7.0, random_seed=42)
    print("\nTest Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
