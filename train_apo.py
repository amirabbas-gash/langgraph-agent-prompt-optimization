import asyncio
import json
import os
from dotenv import load_dotenv
import agentlightning as agl
from openai import AsyncOpenAI  # Still needed for the APO Algorithm initialization

# Import our custom logic
from src.agent import langgraph_agent
from src.grader import calculate_reward

load_dotenv()

# THE ROLLOUT WRAPPER
@agl.rollout
async def langgraph_rollout(task: dict, prompt_template: agl.PromptTemplate) -> float:
    # 1. APO might invent placeholders. We use a defaultdict to prevent KeyErrors.
    from collections import defaultdict
    
    # Create a dictionary of values we actually have
    format_values = {
        "task_input": task["question"],
        "question": task["question"], # Just in case it uses 'question'
    }
    
    # Wrap it so it returns an empty string for any key it doesn't recognize
    safe_values = defaultdict(str, format_values)
    
    # Now format. This will NOT crash if the LLM adds {constraints}
    try:
        # Use simple string formatting to handle the defaultdict
        current_system_prompt = prompt_template.template.format_map(safe_values)
    except Exception:
        # Fallback if the LLM generated totally broken braces
        current_system_prompt = prompt_template.template
    
    # 2. Invoke the LangGraph agent
    inputs = {
        "input": task["question"],
        "system_prompt": current_system_prompt
    }
    result = await langgraph_agent.ainvoke(inputs)
    
    # 3. Grade
    return await calculate_reward(result["output"], task["answer"])


def main():
    # 1. Initialize APO Algorithm 
    # (The APO class requires this client to run the internal 'Textual Gradient' logic)
    openai_engine_client = AsyncOpenAI(api_key=os.getenv("API_KEY"),
                                       base_url=os.getenv("BASE_URL"))
    
    algo = agl.APO(
        openai_engine_client,
        val_batch_size=2,
        gradient_batch_size=2, 
        beam_width=2,
        beam_rounds=2
    )
    remote_store = agl.LightningStoreClient("http://localhost:4747")
    strategy = agl.ClientServerExecutionStrategy(
        role="both", 
        managed_store=False
    )
    # 2. Configure the Trainer
    trainer = agl.Trainer(
        algorithm=algo,
        n_runners=4,
        store=remote_store,
        strategy=strategy,
        initial_resources={
            "prompt_template": agl.PromptTemplate(
                template="You are a specialized AI agent. Task: {task_input}",
                engine="f-string"
            )
        },
        adapter=agl.TraceToMessages() 
    )

    # 3. Load Dataset from data/tasks.jsonl
    if not os.path.exists("data/tasks.jsonl"):
        print("Error: data/tasks.jsonl not found.")
        return

    with open("data/tasks.jsonl", "r") as f:
        dataset = [json.loads(line) for line in f]
    
    train_ds = dataset[:len(dataset)//2]
    val_ds = dataset[len(dataset)//2:]

    # 4. Start Optimization
    print("🚀 Starting Automatic Prompt Optimization (APO)...")
    trainer.fit(
        agent=langgraph_rollout,
        train_dataset=train_ds,
        val_dataset=val_ds
    )

    try:
        best_prompt_template = algo.get_best_prompt()
        print(best_prompt_template.template)
    except ValueError as e:
        print(f"Could not retrieve best prompt: {e}")
        
    input("Press Enter to shut down the server and exit...")
    
if __name__ == "__main__":
    main()