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
        stage_name: Name of the stage (e.g., 'stage_4')
        language: Language code ('en' for English, 'zh' for Chinese)
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__), '..', 'prompts',
        f"{stage_name}_{language}.txt"
    )
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def validate_stage4_output(data: dict) -> tuple[bool, str]:
    """Validate the structure and content of stage 4 output.
    
    Args:
        data: Output data to validate
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ['visual_imagery', 'visual_tags', 'visual_caption']

    # Check required fields
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    # Validate types
    if not isinstance(data['visual_imagery'], str):
        return False, "Field 'visual_imagery' must be a string"

    if not isinstance(data['visual_tags'], str):
        return False, "Field 'visual_tags' must be a string"

    if not isinstance(data['visual_caption'], str):
        return False, "Field 'visual_caption' must be a string"

    # Validate content
    if not data['visual_imagery'].strip():
        return False, "Field 'visual_imagery' cannot be empty"

    if not data['visual_tags'].strip():
        return False, "Field 'visual_tags' cannot be empty"

    if not data['visual_caption'].strip():
        return False, "Field 'visual_caption' cannot be empty"

    return True, ""


def run_stage_4(
    stage_3_output: dict,
    model_name: str = 'gemini',
    log_dir: str = None,
    language: str = 'zh',
    max_retries: int = 3,
    llm_config=None
):
    """
    Executes Stage 4: Narrative Imagery Generation.

    Takes music and caption data and returns visual imagery descriptions.
    
    Args:
        stage_3_output: Output dictionary from stage 3 containing music and caption data
        model_name: LLM model to use ('gemini' or 'mock')
        log_dir: Directory for logging outputs
        language: Language code ('en' or 'zh')
        max_retries: Maximum number of retries on failure
    
    Returns:
        Dictionary with visual_imagery, visual_tags, and visual_caption
    """
    print("\n--- Running Stage 4: Narrative Imagery Generation ---")
    print(
        f"Input: Stage 3 output with caption='{stage_3_output.get('caption_full', 'N/A')[:50]}...'"
    )

    # Load prompt template
    prompt_template = load_prompt("stage_4", language=language)

    # Replace placeholder with stage 3 output
    prompt = prompt_template.replace(
        "{{ENHANCED_STAGE3_OUTPUT}}",
        json.dumps(stage_3_output, indent=2, ensure_ascii=False)
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
            is_valid, error_msg = validate_stage4_output(response_data)
            if not is_valid:
                raise ValueError(f"Output validation failed: {error_msg}")

            # Log the output
            if log_dir:
                with open(
                    os.path.join(log_dir, 'stage_4_output.json'),
                    'w',
                    encoding='utf-8'
                ) as f:
                    json.dump(response_data, f, indent=4, ensure_ascii=False)

            print(f"Successfully generated visual imagery")
            print(
                f"Visual imagery: {response_data['visual_imagery'][:100]}..."
            )
            print(f"Visual tags: {response_data['visual_tags']}")
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
    error_msg = f"Stage 4 failed after {max_retries} attempts. Last error: {last_error}"
    print(f"ERROR: {error_msg}")
    raise RuntimeError(error_msg)


if __name__ == "__main__":
    """Quick test of stage 4 visual imagery generation"""
    import tempfile

    # Sample stage 3 output for testing
    sample_stage_3_output = {
        "theme": "cat",
        "emotion": "Amusing",
        "valence": 7.5,
        "arousal": 6.0,
        "genre": "Jazz",
        "lead_instruments": ["Piano", "Saxophone"],
        "supporting_instruments": ["Double bass", "Drum kit"],
        "tempo": 120,
        "key": "C Major",
        "composition_notes": "一首轻松愉快的爵士乐曲，使用钢琴和萨克斯风作为主奏乐器。",
        "caption_full": "一首俏皮又有趣的爵士四重奏，描绘了一只聪明小猫的午后冒险。",
        "caption_tags": "Jazz, Swing, Playful, Amusing, Piano, Saxophone"
    }

    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with sample inputs
        result = run_stage_4(
            stage_3_output=sample_stage_3_output,
            model_name='gemini',
            log_dir=temp_dir,
            language='zh'
        )
        print("\nTest Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
