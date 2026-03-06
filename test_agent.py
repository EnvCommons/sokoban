import asyncio
import json
import os
from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def test_with_openai():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "sokobanenvironment"
    SPLIT = "test"
    BASE_URL = "http://localhost:8080"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    environment = await or_client.environments.get(name=ENV_NAME, base_url=BASE_URL)
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")
    print(f"Testing with {MODEL_NAME}\n")

    for task in tasks[:1]:  # Test first task
        print(f"Testing task: {task['id']}")
        async with environment.session(
            task=task,
            secrets={"openai_api_key": OPENAI_API_KEY}
        ) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False
            turn = 0
            max_turns = 50

            while not finished and turn < max_turns:
                turn += 1
                print(f"\n--- Turn {turn} ---")

                # Use responses.create(), NOT chat.completions.create()
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list  # Use 'input', NOT 'messages'
                )

                # Response has 'output', NOT 'choices'
                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        print(f"Tool: {item.name}")
                        print(f"Args: {item.arguments}")

                        tool_result = await session.call_tool(
                            item.name,
                            json.loads(str(item.arguments))
                        )

                        finished = tool_result.finished

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text
                        })

                        print(f"Reward: {tool_result.reward:.3f}")
                        print(f"Output: {tool_result.blocks[0].text[:200]}...")

                        if tool_result.finished:
                            print('\n=== FINISHED ===')
                            print(f"Final reward: {tool_result.reward:.3f}")
                            break

            if not finished:
                print(f"\n=== Max turns ({max_turns}) reached without completion ===")


if __name__ == "__main__":
    asyncio.run(test_with_openai())
