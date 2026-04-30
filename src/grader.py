from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

async def calculate_reward(output: str, expected: str) -> float:
    """A highly critical, zero-tolerance LLM judge."""
    
    llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            temperature=0) # Temperature 0 is vital for consistency
    
    # We use a SystemMessage to set the "Hard Critic" persona
    system_msg = SystemMessage(content="""
    You are an extremely harsh and pedantic evaluator. 
    Your goal is to find errors. You prioritize strict adherence to constraints over general helpfulness.
    
    SCORING RULES:
    1. ZERO TOLERANCE: If the user provided specific constraints (e.g., 'No letter E', 'Exactly 3 sentences', 'Format as JSON') and the Assistant failed ANY of them, the score MUST be 0.0, even if the content is otherwise perfect.
    2. FACTUALITY: If the answer is factually incorrect compared to the Expected Answer, the score is 0.0.
    3. PARTIAL CREDIT: Only give partial credit (0.1 - 0.5) if the answer is factually correct but has minor formatting issues.
    4. PERFECT SCORE: Only a 1.0 is given if the output is flawless, follows all constraints, and matches the expected result perfectly.
    """)

    user_content = f"""
    ### EVALUATION TASK
    Expected Reference Answer: {expected}
    Actual Assistant Output: {output}

    ### INSTRUCTIONS
    1. Review the output for any hidden errors or missed instructions.
    2. Be skeptical. If the output 'looks' right but violates a constraint, it is a failure.
    3. Return ONLY a single numeric float between 0.0 and 1.0. 
    """

    try:
        # We pass the system message to ensure the critic persona is locked in
        response = await llm.ainvoke([
            system_msg,
            HumanMessage(content=user_content)
        ])
        
        # Clean the output in case the LLM adds markdown or spaces
        score_str = response.content.strip().replace('`', '').replace('score:', '')
        score = float(score_str)
        
        # Sanity check: Ensure score is within bounds
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        print(f"Error in strict grading: {e}")
        return 0.0