import json
import os
import re
import sys

if __name__ == "__main__":
    sys.path.append('.')

# Try relative import first, fallback to absolute
try:
    from .llm_client import call_llm
except ImportError:
    from va_to_caption.pipeline.llm_client import call_llm

try:
    from ..utils.general import extract_json_from_response
except ImportError:
    from va_to_caption.utils.general import extract_json_from_response


def load_prompt(stage_name, language='en'):
    """Loads a prompt template from the prompts directory.
    
    Args:
        stage_name: Name of the stage (e.g., 'stage_2')
        language: Language code ('en' for English, 'zh' for Chinese)
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__), '..', 'prompts',
        f"{stage_name}_{language}.txt"
    )
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_all_instrument_names(instrument_data: dict) -> set:
    """Recursively extract all instrument names from AudioSet hierarchy.
    
    Args:
        instrument_data: Dictionary containing instrument hierarchy
    
    Returns:
        Set of all instrument names
    """
    names = set()

    if isinstance(instrument_data, dict):
        # Add the name if it exists
        if 'name' in instrument_data:
            names.add(instrument_data['name'])

        # Recursively process children
        if 'children' in instrument_data:
            children = instrument_data['children']
            if isinstance(children, dict):
                for child in children.values():
                    names.update(extract_all_instrument_names(child))

    return names


def extract_all_genre_names(genre_data: dict) -> set:
    """Recursively extract all genre names from FMA hierarchy.
    
    Args:
        genre_data: Dictionary containing genre hierarchy
    
    Returns:
        Set of all genre names
    """
    names = set()

    if isinstance(genre_data, dict):
        for key, value in genre_data.items():
            if isinstance(value, dict):
                # Add the title if it exists
                if 'title' in value:
                    names.add(value['title'])

                # Recursively process children
                if 'children' in value:
                    names.update(extract_all_genre_names(value['children']))

    return names


def load_knowledge_base():
    """Load knowledge bases from JSON files.
    
    Returns:
        Tuple of (musical_instruments, fma_genres, valid_instruments, valid_genres)
    """
    # Load AudioSet Musical Instrument ontology
    audioset_path = os.path.join(
        os.path.dirname(__file__), '..', 'knowledge_base',
        'audioset_music_ontology.json'
    )
    with open(audioset_path, 'r', encoding='utf-8') as f:
        audioset_data = json.load(f)

    # Extract Musical instrument section
    musical_instruments = audioset_data['children']['Musical instrument']

    # Extract all valid instrument names
    valid_instruments = extract_all_instrument_names(musical_instruments)

    # Load FMA genres hierarchy
    fma_path = os.path.join(
        os.path.dirname(__file__), '..', 'knowledge_base',
        'FMA_genres_hierarchy.json'
    )
    with open(fma_path, 'r', encoding='utf-8') as f:
        fma_data = json.load(f)

    # Extract all valid genre names
    valid_genres = extract_all_genre_names(fma_data)

    return musical_instruments, fma_data, valid_instruments, valid_genres


def validate_stage2_output(
    data: dict,
    valid_genres: set = None,
    valid_instruments: set = None
) -> tuple[bool, str]:
    """Validate the structure and content of stage 2 output.
    
    Args:
        data: Output data to validate
        valid_genres: Set of valid genre names from FMA knowledge base
        valid_instruments: Set of valid instrument names from AudioSet knowledge base
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = [
        'genre', 'lead_instruments', 'supporting_instruments', 'tempo', 'key',
        'composition_notes'
    ]

    # Check required fields
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    # Validate types
    if not isinstance(data['genre'], str):
        return False, "Field 'genre' must be a string"

    if not isinstance(data['lead_instruments'], list):
        return False, "Field 'lead_instruments' must be a list"

    if not isinstance(data['supporting_instruments'], list):
        return False, "Field 'supporting_instruments' must be a list"

    if not isinstance(data['tempo'], int):
        return False, "Field 'tempo' must be an integer"

    if not isinstance(data['key'], str):
        return False, "Field 'key' must be a string"

    if not isinstance(data['composition_notes'], str):
        return False, "Field 'composition_notes' must be a string"

    # Validate content
    if not data['genre'].strip():
        return False, "Field 'genre' cannot be empty"

    if len(data['lead_instruments']) == 0:
        return False, "Field 'lead_instruments' cannot be empty"

    if len(data['supporting_instruments']) == 0:
        return False, "Field 'supporting_instruments' cannot be empty"

    if data['tempo'] <= 0:
        return False, "Field 'tempo' must be positive"

    if not data['key'].strip():
        return False, "Field 'key' cannot be empty"

    if not data['composition_notes'].strip():
        return False, "Field 'composition_notes' cannot be empty"

    # Validate genre against knowledge base
    if valid_genres is not None:
        genre = data['genre'].strip()
        if genre not in valid_genres:
            return False, f"Genre '{genre}' not found in FMA knowledge base. Must be one of the valid genres."

    # Validate instruments against knowledge base
    if valid_instruments is not None:
        # Check lead instruments
        for instrument in data['lead_instruments']:
            if instrument.strip() not in valid_instruments:
                return False, f"Lead instrument '{instrument}' not found in AudioSet knowledge base. Must be one of the valid instruments."

        # Check supporting instruments
        for instrument in data['supporting_instruments']:
            if instrument.strip() not in valid_instruments:
                return False, f"Supporting instrument '{instrument}' not found in AudioSet knowledge base. Must be one of the valid instruments."

    return True, ""


def run_stage_2(
    theme: str,
    emotion: str,
    valence: float,
    arousal: float,
    model_name: str = 'gemini',
    log_dir: str = None,
    language: str = 'zh',
    innovation: str = 'medium',
    max_retries: int = 3,
    llm_config=None
):
    """
    Executes Stage 2: Music Content Association.

    Takes emotional mapping from stage 1 and returns music composition parameters.
    
    Args:
        theme: Theme keyword from stage 1
        emotion: Emotion label from stage 1
        valence: Valence value [1-9]
        arousal: Arousal value [1-9]
        model_name: LLM model to use ('gemini' or 'mock')
        log_dir: Directory for logging outputs
        language: Language code ('en' or 'zh')
        innovation: Innovation level ('low', 'medium', 'high')
        max_retries: Maximum number of retries on failure
    
    Returns:
        Dictionary with music composition parameters
    """
    print(f"\n--- Running Stage 2: Music Content Association ---")
    print(
        f"Input: Theme='{theme}', Emotion='{emotion}', V={valence}, A={arousal}, Innovation={innovation}"
    )

    # Load knowledge bases
    musical_instruments, fma_genres, valid_instruments, valid_genres = load_knowledge_base(
    )

    # Load prompt template
    prompt_template = load_prompt("stage_2", language=language)

    # Replace placeholders
    prompt = prompt_template.replace(
        "{{AUDIOSET}}",
        json.dumps(musical_instruments, indent=2, ensure_ascii=False)
    )
    prompt = prompt.replace(
        "{{FMA}}", json.dumps(fma_genres, indent=2, ensure_ascii=False)
    )
    prompt = prompt.replace("{{THEME}}", json.dumps(theme, ensure_ascii=False))
    prompt = prompt.replace(
        "{{EMOTION}}", json.dumps(emotion, ensure_ascii=False)
    )
    prompt = prompt.replace("{{VALENCE}}", str(valence))
    prompt = prompt.replace("{{AROUSAL}}", str(arousal))
    prompt = prompt.replace(
        "{{INNOVATION}}", json.dumps(innovation, ensure_ascii=False)
    )

    # Retry loop
    last_error = None
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")

            # Call the LLM
            response_str = call_llm(model_name, prompt, config=llm_config)

            # Extract JSON from response
            json_str = extract_json_from_response(response_str)

            # Parse the JSON response
            response_data = json.loads(json_str)

            # Validate the output
            is_valid, error_msg = validate_stage2_output(
                response_data,
                valid_genres=valid_genres,
                valid_instruments=valid_instruments
            )
            if not is_valid:
                raise ValueError(f"Output validation failed: {error_msg}")

            # Log the output
            if log_dir:
                with open(
                    os.path.join(log_dir, 'stage_2_output.json'),
                    'w',
                    encoding='utf-8'
                ) as f:
                    json.dump(response_data, f, indent=4, ensure_ascii=False)

            print(
                f"Successfully generated music composition: Genre={response_data['genre']}, Tempo={response_data['tempo']} BPM"
            )
            return response_data

        except json.JSONDecodeError as e:
            last_error = f"JSON parsing error: {e}"
            print(f"  {last_error}")
            if attempt < max_retries - 1:
                print(f"  Retrying...")
        except ValueError as e:
            last_error = f"Validation error: {e}"
            print(f"  {last_error}")
            if attempt < max_retries - 1:
                print(f"  Retrying...")
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            print(f"  {last_error}")
            if attempt < max_retries - 1:
                print(f"  Retrying...")

    # All retries failed
    error_msg = f"Stage 2 failed after {max_retries} attempts. Last error: {last_error}"
    print(f"ERROR: {error_msg}")
    raise RuntimeError(error_msg)


if __name__ == "__main__":
    """Quick test of stage 2 mapping"""
    import tempfile

    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with sample inputs
        result = run_stage_2(
            theme="cat",
            emotion="Amusing",
            valence=7.5,
            arousal=6.0,
            model_name='gemini_xi',
            log_dir=temp_dir,
            language='zh',
            innovation='medium'
        )
        print("\nTest Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
