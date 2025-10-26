import asyncio
import os
import sys
import json
import base64

import httpx
from computer_use_demo.loop import sampling_loop, APIProvider
from computer_use_demo.tools import ToolResult


async def main():
    # Set up your Anthropic API key and model
    api_key = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
    if not api_key or api_key.startswith("YOUR_API_KEY"):
        raise ValueError(
            "Please first set your API key in the ANTHROPIC_API_KEY environment variable"
        )
    provider = APIProvider.ANTHROPIC

    # Check if the instruction is provided via command line arguments
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:])
    else:
        instruction = "Save an image of a cat to the desktop."

    print(
        f"Starting Claude 'Computer Use' with Claude 4.5 Sonnet (claude-sonnet-4-5-20250929).\nPress ctrl+c to stop.\nInstructions provided: '{instruction}'"
    )

    # Set up the initial messages
    messages = [
        {
            "role": "user",
            "content": instruction,
        }
    ]

    # Define callbacks (you can customize these)
    def output_callback(content_block):
        if isinstance(content_block, dict) and content_block.get("type") == "text":
            print("Assistant:", content_block.get("text"))
        elif isinstance(content_block, dict) and content_block.get("type") == "thinking":
            print("Thinking:", content_block.get("thinking", "")[:100], "...")

    def tool_output_callback(result: ToolResult, tool_use_id: str):
        if result.output:
            print(f"> Tool Output [{tool_use_id}]:", result.output)
        if result.error:
            print(f"!!! Tool Error [{tool_use_id}]:", result.error)
        if result.base64_image:
            # Save the image to a file if needed
            os.makedirs("screenshots", exist_ok=True)
            image_data = result.base64_image
            with open(f"screenshots/screenshot_{tool_use_id}.png", "wb") as f:
                f.write(base64.b64decode(image_data))
            print(f"Took screenshot screenshot_{tool_use_id}.png")

    def api_response_callback(
        request: httpx.Request,
        response: httpx.Response | object | None,
        error: Exception | None,
    ):
        if error:
            print(f"\n!!! API Error: {error}\n")
        elif response and isinstance(response, httpx.Response):
            try:
                response_json = response.json()
                if "content" in response_json:
                    print(
                        "\n---------------\nAPI Response:\n",
                        json.dumps(response_json["content"], indent=4),
                        "\n",
                    )
            except Exception:
                pass

    # Run the sampling loop with Claude 4.5 Sonnet
    messages = await sampling_loop(
        model="claude-sonnet-4-5-20250929",
        provider=provider,
        system_prompt_suffix="",
        messages=messages,
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
        api_response_callback=api_response_callback,
        api_key=api_key,
        only_n_most_recent_images=10,
        max_tokens=16384,  # Claude 4.5 Sonnet supports higher token limits
        tool_version="computer_use_20250124",  # Use the latest tool version
        thinking_budget=None,  # Optional: set to enable extended thinking
        token_efficient_tools_beta=False,  # Optional: enable token efficient tools
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Encountered Error:\n{e}")
