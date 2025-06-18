import os
import time
import runpod
from vllm import LLM, SamplingParams

# Global variables
llm = None


def initialize_llm(input_data):
    global llm

    if not llm:
        print("Initializing VLLM...")
        start_time = time.time()

        # Hardcoded VLLM engine arguments
        engine_args = {
            "model": "/workspace/models/Mistral-Small",  # !! Falls ich was ändere in Dockefile, hier auch ändern !!!
            "trust_remote_code": True,
            "enforce_eager": True,
            "quantization": "compressed-tensors",
            "max_model_len": 8192,
        }

        # Override with input args if provided (allows runtime customization)
        if "engine_args" in input_data:
            engine_args.update(input_data["engine_args"])

        print(f"Engine args: {engine_args}")  # Debug print

        llm = LLM(**engine_args)
        print('─' * 20,
              "--- VLLM engine and model cold start initialization took %s seconds ---" % (time.time() - start_time),
              '─' * 20, "*Unfortunately, this time is being billed.", sep=os.linesep)


def process_batch_requests(batch_data):
    """Convert batch requests to text prompts"""
    prompts = []

    for request in batch_data:
        if "messages" in request:
            # Handle chat format - convert to single prompt
            messages = request["messages"]
            text_parts = []

            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")

                if isinstance(content, list):
                    # Extract text from multimodal content
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(f"{role}: {item['text']}")
                else:
                    # Simple text content
                    text_parts.append(f"{role}: {content}")

            # Combine all messages into a single prompt
            combined_prompt = "\n".join(text_parts)
            prompts.append(combined_prompt)

        elif "prompt" in request:
            # Handle simple prompt format
            prompts.append(str(request["prompt"]))
        else:
            # Fallback - try to extract any text content
            prompts.append(str(request))

    return prompts


def create_sampling_params(batch_data):
    """Create sampling parameters from first request"""
    first_request = batch_data[0] if batch_data else {}

    params = {
        "max_tokens": first_request.get("max_tokens", 1024),
        "temperature": first_request.get("temperature", 0.7),
        "top_p": first_request.get("top_p", 1.0),
        "top_k": first_request.get("top_k", -1),
        "repetition_penalty": first_request.get("repetition_penalty", 1.0),
        "stop": first_request.get("stop", None),
    }

    # Remove None values to avoid errors
    params = {k: v for k, v in params.items() if v is not None}

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

        # Process requests to extract text prompts
        prompts = process_batch_requests(batch_requests)

        # Debug: Print processed prompts
        print(f"Processing {len(prompts)} text prompts")
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i}: {prompt[:100]}...")

        # Create sampling parameters
        sampling_params = create_sampling_params(batch_requests)

        # Generate responses
        print(f"Generating responses...")
        outputs = llm.generate(prompts, sampling_params)

        # Format results
        results = []
        for i, output in enumerate(outputs):
            result = {
                "index": i,
                "text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
                "prompt_tokens": len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else None,
                "completion_tokens": len(output.outputs[0].token_ids) if hasattr(output.outputs[0],
                                                                                 'token_ids') else None
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