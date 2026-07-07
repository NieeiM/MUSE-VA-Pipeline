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
        stage_name: Name of the stage (e.g., 'stage_5')
        language: Language code ('en' for English, 'zh' for Chinese)
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__), '..', 'prompts',
        f"{stage_name}_{language}.txt"
    )
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def validate_stage5_output(data: dict) -> tuple[bool, str]:
    """Validate the structure and content of stage 5 output.
    
    Args:
        data: Output data to validate
    
    Returns:
        (is_valid, error_message)
    """
    # Check for consistency field
    if 'consistency' not in data:
        return False, "Missing required field: consistency"

    consistency = data['consistency']

    # Check required fields in consistency
    if not isinstance(consistency, dict):
        return False, "Field 'consistency' must be a dictionary"

    if 'result' not in consistency:
        return False, "Missing required field: consistency.result"

    if 'reason' not in consistency:
        return False, "Missing required field: consistency.reason"

    # Validate types
    if not isinstance(consistency['result'], bool):
        return False, "Field 'consistency.result' must be a boolean"

    if not isinstance(consistency['reason'], str):
        return False, "Field 'consistency.reason' must be a string"

    # Validate content
    if not consistency['reason'].strip():
        return False, "Field 'consistency.reason' cannot be empty"

    return True, ""


def run_stage_5(
    stage_4_output: dict,
    model_name: str = 'gemini',
    log_dir: str = None,
    language: str = 'zh',
    max_retries: int = 3,
    llm_config=None
):
    """
    Executes Stage 5: Consistency Check.

    Takes combined music and visual data and checks for consistency.
    
    Args:
        stage_4_output: Output dictionary from stage 4 containing all accumulated data
        model_name: LLM model to use ('gemini' or 'mock')
        log_dir: Directory for logging outputs
        language: Language code ('en' or 'zh')
        max_retries: Maximum number of retries on failure
    
    Returns:
        Dictionary with consistency check results
    """
    print("\n--- Running Stage 5: Consistency Check ---")
    print(f"Input: Checking consistency of full pipeline output")

    # Load prompt template
    prompt_template = load_prompt("stage_5", language=language)

    # Replace placeholder with stage 4 output
    prompt = prompt_template.replace(
        "{{ENHANCED_STAGE4_OUTPUT}}",
        json.dumps(stage_4_output, indent=2, ensure_ascii=False)
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
            is_valid, error_msg = validate_stage5_output(response_data)
            if not is_valid:
                raise ValueError(f"Output validation failed: {error_msg}")

            # Log the output
            if log_dir:
                with open(
                    os.path.join(log_dir, 'stage_5_output.json'),
                    'w',
                    encoding='utf-8'
                ) as f:
                    json.dump(response_data, f, indent=4, ensure_ascii=False)

            consistency_result = response_data['consistency']['result']
            consistency_reason = response_data['consistency']['reason']

            result_label = "PASS" if consistency_result else "FAIL"
            print(f"Consistency check {result_label}: {consistency_result}")
            print(f"Reason: {consistency_reason[:100]}...")

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
    error_msg = f"Stage 5 failed after {max_retries} attempts. Last error: {last_error}"
    print(f"ERROR: {error_msg}")
    raise RuntimeError(error_msg)


if __name__ == "__main__":
    """Quick test of stage 5 consistency check"""
    import tempfile

    # Sample stage 4 output for testing (includes all previous stage data)
    sample_stage_4_output = {
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
            "一首轻松愉快的爵士乐曲。",
        "caption_full":
            "一首俏皮又有趣的爵士四重奏，描绘了一只聪明小猫的午后冒险。",
        "caption_tags":
            "Jazz, Swing, Playful, Amusing, Piano, Saxophone",
        "visual_imagery":
            "A playful ginger cat prancing through a sun-dappled jazz club in the afternoon.",
        "visual_tags":
            "cat, jazz club, playful, afternoon, warm lighting, whimsical",
        "visual_caption":
            "A whimsical scene of a curious orange cat exploring a cozy jazz club bathed in golden afternoon light"
    }

    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with sample inputs
        result = run_stage_5(
            stage_4_output=sample_stage_4_output,
            model_name='gemini',
            log_dir=temp_dir,
            language='zh'
        )
        print("\nTest Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
