import asyncio
from env import SokobanEnvironment, MoveParams


async def test_locally():
    print("=== Starting Sokoban smoke test ===\n")

    # Test 1: List tasks
    print("Test 1: Listing tasks...")
    tasks = SokobanEnvironment.list_tasks(split="test")
    print(f"Found {len(tasks)} tasks")
    print(f"First task: {tasks[0]}\n")

    example_task = tasks[0]

    # Test 2: Create environment and get prompt
    print("Test 2: Getting prompt...")
    env = SokobanEnvironment(task_spec=example_task)
    prompt = await env.get_prompt()
    prompt_text = prompt[0].text
    print(f"Prompt (first 500 chars): {prompt_text[:500]}...\n")

    # Test 3: Call move tool
    print("Test 3: Calling move tool...")
    move_params = MoveParams(direction="right")
    result = await env.move(move_params)
    print(f"Reward: {result.reward}")
    print(f"Finished: {result.finished}")
    print(f"Output (first 500 chars): {result.blocks[0].text[:500]}")
    print(f"Metadata: {result.metadata}\n")

    print("=== Smoke test PASSED ===")


if __name__ == "__main__":
    asyncio.run(test_locally())
