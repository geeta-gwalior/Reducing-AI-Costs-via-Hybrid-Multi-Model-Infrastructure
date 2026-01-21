import sys
import os

# Ensure we can import from 'agents' folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from agents.agent import KifayatiRouter

def main():
    print("---  Kifayati Agent System Initialized ---")
    print("--- Type 'exit' to quit ---\n")
    
    # Initialise the Router
    router = KifayatiRouter()
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Bye!")
            break
        
        try:
            response = router.route_and_execute(user_input)
            print(f"Agent: {response}")
        except Exception as e:
            print(f" Error: {e}")
            print("Tip: Check if Endpoint ID in .env is correct and Active.")

if __name__ == "__main__":
    main()
