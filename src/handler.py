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

        # Model should be set via VLLM_MODEL environment variable in Dockerfile

        # Set other defaults if not provided
        if "trust_remote_code" not in engine_args:
            engine_args["trust_remote_code"] = True

        llm = LLM(**engine_args)
        print('─' * 20, "--- VLLM engine and model cold start initialization took %s seconds ---" % (time.time() - start_time), '─' * 20, "*Unfortunately, this time is being billed.", sep=os.linesep)

def process_batch_requests(batch_data):
    """Convert batch requests to vLLM format"""
    prompts = []

    for request in batch_data:
        if "messages" in request:
            # Handle chat format
            messages = request["messages"]
            prompt = []

            for message in messages:
                if message.get("role") == "user":
                    content = message.get("content", "")

                    if isinstance(content, list):
                        # Multimodal content (text + images)
                        for item in content:
                            if item.get("type") == "text":
                                prompt.append(item["text"])
                            elif item.get("type") == "image_url":
                                prompt.append({"type": "image", "image": item["image_url"]["url"]})
                    else:
                        # Simple text
                        prompt.append(content)

            prompts.append(prompt)

        elif "prompt" in request:
            # Handle simple prompt format
            prompts.append(request["prompt"])

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


async def handler(job):
    try:
        input_data = job["input"]

        # Initialize LLM
        initialize_llm(input_data)

        # Handle prewarm
        if "prewarm" in input_data:
            yield {"warm": True}
            return

        # Get batch requests
        if "batch" not in input_data:
            yield {"error": "Expected 'batch' key with list of requests"}
            return

        batch_requests = input_data["batch"]

        # Process requests
        prompts = process_batch_requests(batch_requests)
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

        yield {"results": results}

    except Exception as e:
        yield {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})