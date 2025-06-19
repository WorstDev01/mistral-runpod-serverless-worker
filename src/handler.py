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
            "tokenizer_mode": "mistral",  # Important for Mistral Small 3.1
        }

        # Override with input args if provided (allows runtime customization)
        if "engine_args" in input_data:
            engine_args.update(input_data["engine_args"])

        print(f"Engine args: {engine_args}")  # Debug print

        llm = LLM(**engine_args)
        print('─' * 20,
              "--- VLLM engine and model cold start initialization took %s seconds ---" % (time.time() - start_time),
              '─' * 20, "*Unfortunately, this time is being billed.", sep=os.linesep)


def format_chat_messages(messages):
    """Format chat messages for Mistral Small 3.1 using V7-Tekken template"""
    system_prompt = ""
    conversation_parts = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        # Handle multimodal content (list format)
        if isinstance(content, list):
            text_content = ""
            for item in content:
                if item.get("type") == "text":
                    text_content += item.get("text", "")
            content = text_content

        if role == "system":
            system_prompt = content
        elif role == "user":
            conversation_parts.append(("user", content))
        elif role == "assistant":
            conversation_parts.append(("assistant", content))

    # Build the formatted prompt using V7-Tekken template
    formatted_prompt = "<s>"

    # Add system prompt if exists
    if system_prompt:
        formatted_prompt += f"[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT]"

    # Add conversation
    for i, (role, content) in enumerate(conversation_parts):
        if role == "user":
            formatted_prompt += f"[INST]{content}[/INST]"
            # If this is the last message and it's from user, don't add assistant response
            if i == len(conversation_parts) - 1:
                break
        elif role == "assistant":
            formatted_prompt += content + "</s>"

    return formatted_prompt


def process_batch_requests(batch_data):
    """Convert batch requests to text prompts with proper formatting"""
    prompts = []

    for request in batch_data:
        if "messages" in request:
            # Handle chat format - use proper Mistral formatting
            messages = request["messages"]
            formatted_prompt = format_chat_messages(messages)
            prompts.append(formatted_prompt)

            # Debug: Print the formatted prompt
            print(f"Formatted prompt: {formatted_prompt[:200]}...")

        elif "prompt" in request:
            # Handle simple prompt format
            prompts.append(str(request["prompt"]))
        else:
            # Fallback - try to extract any text content
            prompts.append(str(request))

    return prompts


def create_sampling_params(batch_data):
    """Create sampling parameters from first request - only use provided params"""
    first_request = batch_data[0] if batch_data else {}

    # Only include parameters that are explicitly provided
    params = {}

    if "max_tokens" in first_request:
        params["max_tokens"] = first_request["max_tokens"]
    if "temperature" in first_request:
        params["temperature"] = first_request["temperature"]
    if "top_p" in first_request:
        params["top_p"] = first_request["top_p"]
    if "top_k" in first_request:
        params["top_k"] = first_request["top_k"]
    if "repetition_penalty" in first_request:
        params["repetition_penalty"] = first_request["repetition_penalty"]
    if "stop" in first_request:
        params["stop"] = first_request["stop"]

    print(f"Sampling params (only provided): {params}")  # Debug print

    return SamplingParams(**params) if params else SamplingParams()


def handler(job):
    try:
        input_data = job["input"]

        print(f"Received input data: {input_data}")  # Debug print

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

        print(f"Processing {len(batch_requests)} batch requests")

        # Process requests to extract text prompts
        prompts = process_batch_requests(batch_requests)

        # Debug: Print processed prompts
        print(f"Processing {len(prompts)} text prompts")
        for i, prompt in enumerate(prompts):
            print(f"Full prompt {i}: {prompt}")

        # Create sampling parameters
        sampling_params = create_sampling_params(batch_requests)

        # Generate responses
        print(f"Generating responses with sampling params: {sampling_params}")
        outputs = llm.generate(prompts, sampling_params)

        # Format results
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            print(f"Generated text {i}: '{generated_text}'")  # Debug print

            result = {
                "index": i,
                "text": generated_text,
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