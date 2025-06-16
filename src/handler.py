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

        # Get engine args from environment or input
        engine_args = {}

        # Load from environment variables
        for key, value in os.environ.items():
            if key.startswith("VLLM_"):
                param_name = key.replace("VLLM_", "").lower()
                engine_args[param_name] = value

        # Override with input args if provided
        if "engine_args" in input_data:
            engine_args.update(input_data["engine_args"])

        # Set other defaults if not provided
        if "trust_remote_code" not in engine_args:
            engine_args["trust_remote_code"] = True

        llm = LLM(**engine_args)
        print('─' * 20,
              "--- VLLM engine and model cold start initialization took %s seconds ---" % (time.time() - start_time),
              '─' * 20, "*Unfortunately, this time is being billed.", sep=os.linesep)


def process_batch_requests(batch_data):
    """Convert batch requests to vLLM format"""
    prompts = []

    for request in batch_data:
        if "messages" in request:
            # Handle chat format - convert to simple text for now
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
                        # Note: For image handling, you'd need to process images differently
                        # This is a simplified version that only handles text
                else:
                    # Simple text content
                    text_parts.append(f"{role}: {content}")

            # Join all messages into a single prompt string
            prompt_text = "\n".join(text_parts)
            prompts.append(prompt_text)

        elif "prompt" in request:
            # Handle simple prompt format
            prompt = request["prompt"]
            if isinstance(prompt, str):
                prompts.append(prompt)
            else:
                # Convert to string if it's not already
                prompts.append(str(prompt))

    return prompts


def create_sampling_params(batch_data):
    """Create sampling parameters from first request"""
    first_request = batch_data[0] if batch_data else {}

    params = {
        "max_tokens": first_request.get("max_tokens", 100),
        "temperature": first_request.get("temperature", 0.7),
        "top_p": first_request.get("top_p", 1.0),
    }

    return SamplingParams(**params)


def handler(job):  # Removed async - not needed
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
        prompts = process_batch_requests(batch_requests)

        # Debug: Print prompts to see what we're sending to vLLM
        print(f"Processed prompts: {prompts}")

        # Validate prompts are all strings
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                return {"error": f"Prompt at index {i} is not a string: {type(prompt)}"}  #

        sampling_params = create_sampling_params(batch_requests)

        # Generate responses
        print(f"Generating responses for {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params)

        # Format results
        results = []
        for i, output in enumerate(outputs):
            result = {
                "index": i,
                "text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason
            }
            results.append(result)

        print(f"Successfully generated {len(results)} results")  # Debug log
        return {"results": results}

    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
