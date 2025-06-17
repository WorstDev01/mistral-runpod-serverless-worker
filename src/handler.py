import os
import time
import base64
import runpod
from vllm import LLM, SamplingParams
from PIL import Image
import io

# Global variables
llm = None


def initialize_llm(input_data):
    global llm

    if not llm:
        print("Initializing VLLM...")
        start_time = time.time()

        # Get engine args from environment or input
        engine_args = {}

        # Load from environment variables
        for key, value in os.environ.items():
            if key.startswith("VLLM_"):
                param_name = key.replace("VLLM_", "").lower()
                # Special handling for JSON values
                if param_name == "limit_mm_per_prompt":
                    import json
                    try:
                        engine_args[param_name] = json.loads(value)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {key} as JSON: {value}")
                        engine_args[param_name] = value
                else:
                    engine_args[param_name] = value

        # Override with input args if provided
        if "engine_args" in input_data:
            engine_args.update(input_data["engine_args"])

        llm = LLM(**engine_args)
        print('─' * 20,
              "--- VLLM engine and model cold start initialization took %s seconds ---" % (time.time() - start_time),
              '─' * 20, "*Unfortunately, this time is being billed.", sep=os.linesep)


def decode_base64_image(base64_string):
    """Decode base64 image string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]

        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


def process_batch_requests(batch_data):
    """Convert batch requests to vLLM format with proper multimodal handling"""
    vllm_inputs = []

    for request in batch_data:
        if "messages" in request:
            # Handle chat format with multimodal support
            messages = request["messages"]

            for message in messages:
                content = message.get("content", "")

                if isinstance(content, list):
                    # Handle multimodal content
                    text_parts = []
                    images = []

                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item["text"])
                        elif item.get("type") == "image_url":
                            # Extract image data
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "")

                            if url.startswith("data:"):
                                # Handle base64 encoded image
                                image = decode_base64_image(url)
                                if image:
                                    images.append(image)
                            else:
                                print(f"Unsupported image URL format: {url}")

                    # Combine text parts
                    combined_text = " ".join(text_parts)

                    # Create vLLM input format
                    if images:
                        # For vision models, use the proper format
                        vllm_input = {
                            "prompt": f"<image>\n{combined_text}",  # InternVL expects this format
                            "multi_modal_data": {"image": images[0]}  # Single image for now
                        }
                        vllm_inputs.append(vllm_input)
                    else:
                        # Text-only request
                        vllm_inputs.append({"prompt": combined_text})

                else:
                    # Simple text content
                    vllm_inputs.append({"prompt": content})

        elif "prompt" in request:
            # Handle simple prompt format
            vllm_inputs.append({"prompt": str(request["prompt"])})

    return vllm_inputs


def create_sampling_params(batch_data):
    """Create sampling parameters from first request"""
    first_request = batch_data[0] if batch_data else {}

    params = {
        "max_tokens": first_request.get("max_tokens", 1024),
        "temperature": first_request.get("temperature", 0.7),
        "top_p": first_request.get("top_p", 1.0),
    }

    return SamplingParams(**params)


def handler(job):
    try:
        input_data = job["input"]

        # Initialize LLM
        initialize_llm(input_data)

        # Handle prewarm
        if "prewarm" in input_data:
            return {"warm": True}

        # Get batch requests
        if "batch" not in input_data:
            return {"error": "Expected 'batch' key with list of requests"}

        batch_requests = input_data["batch"]

        if not batch_requests:
            return {"error": "Batch requests list is empty"}

        # Process requests
        vllm_inputs = process_batch_requests(batch_requests)

        # Debug: Print processed inputs
        print(f"Processed {len(vllm_inputs)} inputs")
        for i, inp in enumerate(vllm_inputs):
            if "multi_modal_data" in inp:
                print(f"Input {i}: Multimodal - {inp['prompt'][:50]}...")
            else:
                print(f"Input {i}: Text only - {inp['prompt'][:50]}...")

        sampling_params = create_sampling_params(batch_requests)

        # Separate multimodal and text-only inputs
        text_only_prompts = []
        multimodal_inputs = []
        input_mapping = []  # Track which original input each corresponds to

        for i, inp in enumerate(vllm_inputs):
            if "multi_modal_data" in inp:
                multimodal_inputs.append(inp)
                input_mapping.append(("multimodal", len(multimodal_inputs) - 1, i))
            else:
                text_only_prompts.append(inp["prompt"])
                input_mapping.append(("text", len(text_only_prompts) - 1, i))

        # Generate responses
        print(f"Generating responses...")
        all_outputs = [None] * len(vllm_inputs)

        # Process text-only requests
        if text_only_prompts:
            print(f"Processing {len(text_only_prompts)} text-only requests")
            text_outputs = llm.generate(text_only_prompts, sampling_params)

        # Process multimodal requests
        if multimodal_inputs:
            print(f"Processing {len(multimodal_inputs)} multimodal requests")
            mm_prompts = [inp["prompt"] for inp in multimodal_inputs]
            mm_data = [inp["multi_modal_data"] for inp in multimodal_inputs]
            mm_outputs = llm.generate(mm_prompts, sampling_params, multi_modal_data=mm_data)

        # Reconstruct results in original order
        text_idx = 0
        mm_idx = 0

        for input_type, type_idx, original_idx in input_mapping:
            if input_type == "text":
                all_outputs[original_idx] = text_outputs[text_idx]
                text_idx += 1
            else:  # multimodal
                all_outputs[original_idx] = mm_outputs[mm_idx]
                mm_idx += 1

        # Format results
        results = []
        for i, output in enumerate(all_outputs):
            if output is not None:
                result = {
                    "index": i,
                    "text": output.outputs[0].text,
                    "finish_reason": output.outputs[0].finish_reason
                }
                results.append(result)

        print(f"Successfully generated {len(results)} results")
        return {"results": results}

    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})