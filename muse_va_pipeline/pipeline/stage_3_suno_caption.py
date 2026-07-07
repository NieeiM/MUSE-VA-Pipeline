import os
import json
import sys

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
        stage_name: Name of the stage (e.g., 'stage_3')
        language: Language code ('en' for English, 'zh' for Chinese)
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__), '..', 'prompts',
        f"{stage_name}_{language}.txt"
    )
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def validate_stage3_output(data: dict) -> tuple[bool, str]:
    """Validate the structure and content of stage 3 output.
    
    Args:
        data: Output data to validate
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ['caption_full', 'caption_tags']

    # Check required fields
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    # Validate types
    if not isinstance(data['caption_full'], str):
        return False, "Field 'caption_full' must be a string"

    if not isinstance(data['caption_tags'], str):
        return False, "Field 'caption_tags' must be a string"

    # Validate content
    if not data['caption_full'].strip():
        return False, "Field 'caption_full' cannot be empty"

    if not data['caption_tags'].strip():
        return False, "Field 'caption_tags' cannot be empty"

    return True, ""


def run_stage_3(
    stage_2_output: dict,
    model_name: str = 'gemini',
    log_dir: str = None,
    language: str = 'zh',
    max_retries: int = 3,
    llm_config=None
):
    """
    Executes Stage 3: Suno Caption Authoring.

    Takes a music concept and emotional palette and returns a descriptive Suno caption.
    
    Args:
        stage_2_output: Output dictionary from stage 2 containing music composition parameters
        model_name: LLM model to use ('gemini' or 'mock')
        log_dir: Directory for logging outputs
        language: Language code ('en' or 'zh')
        max_retries: Maximum number of retries on failure
    
    Returns:
        Dictionary with caption_full and caption_tags
    """
    print("\n--- Running Stage 3: Suno Caption Authoring ---")
    print(
        f"Input: Stage 2 output with genre='{stage_2_output.get('genre', 'N/A')}'"
    )

    # Load prompt template
    prompt_template = load_prompt("stage_3", language=language)

    # Replace placeholder with stage 2 output
    prompt = prompt_template.replace(
        "{{ENHANCED_STAGE2_OUTPUT}}",
        json.dumps(stage_2_output, indent=2, ensure_ascii=False)
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
            is_valid, error_msg = validate_stage3_output(response_data)
            if not is_valid:
                raise ValueError(f"Output validation failed: {error_msg}")

            # Log the output
            if log_dir:
                with open(
                    os.path.join(log_dir, 'stage_3_output.json'),
                    'w',
                    encoding='utf-8'
                ) as f:
                    json.dump(response_data, f, indent=4, ensure_ascii=False)

            print(f"Successfully generated Suno caption")
            print(f"Caption (full): {response_data['caption_full'][:100]}...")
            print(f"Caption (tags): {response_data['caption_tags']}")
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
    error_msg = f"Stage 3 failed after {max_retries} attempts. Last error: {last_error}"
    print(f"ERROR: {error_msg}")
    raise RuntimeError(error_msg)


if __name__ == "__main__":
    """Quick test of stage 3 caption generation"""
    import tempfile

    # Sample stage 2 output for testing
    sample_stage_2_output = {
        "theme":
            "cat",
        "emotion":
            "Amusing",
        "valence":
            7.5,
        "arousal":
            6.0,
        "genre":
            "Jazz",
        "lead_instruments": ["Piano", "Saxophone"],
        "supporting_instruments": ["Double bass", "Drum kit"],
        "tempo":
            120,
        "key":
            "C Major",
        "composition_notes":
            "一首轻松愉快的爵士乐曲，使用钢琴和萨克斯风作为主奏乐器，配以低音提琴和鼓组，节奏明快，适合表现猫咪的俏皮可爱。"
    }

    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with sample inputs
        result = run_stage_3(
            stage_2_output=sample_stage_2_output,
            model_name='gemini',
            log_dir=temp_dir,
            language='zh'
        )
        print("\nTest Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
