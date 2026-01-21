from google.cloud import aiplatform
from google.cloud import aiplatform_v1
import vertexai
from vertexai.generative_models import GenerativeModel
from .config import PROJECT_ID, LOCATION, GEMMA_ENDPOINT_ID, SIMPLE_KEYWORDS

# Initialise Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- 1. DEFINE AGENT CLASS (ADK Style Wrapper) ---
class LlmAgent:
    def __init__(self, name, model_type, instruction):
        self.name = name
        self.model_type = model_type # 'gemini' or 'gemma'
        self.instruction = instruction
        
        # Setup Gemini
        if model_type == 'gemini':
            self.model = GenerativeModel("gemini-2.5-flash")
            
        # Setup Gemma (Endpoint)
        elif model_type == 'gemma':
            self.endpoint = aiplatform.Endpoint(
                endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{GEMMA_ENDPOINT_ID}"
            )

    def ask(self, query):
        if self.model_type == 'gemini':
            # Gemini Call
            response = self.model.generate_content(f"{self.instruction}\n\nUser Query: {query}")
            return response.text
            
        elif self.model_type == 'gemma':
            # Gemma Endpoint Call
            instances = [{"prompt": f"{self.instruction}\nUser: {query}\nAnswer:", "max_tokens": 200}]
            response = self.endpoint.predict(instances=instances)
            # Parsing Gemma response (adjustment might be needed based on specific container)
            try:
                return response.predictions[0]
            except:
                return str(response.predictions)

# --- 2. THE ROUTER SYSTEM ---
class KifayatiRouter:
    def __init__(self):
        self.cache = {}
        
        # Sub-Agent 1: Gemma (Sasta)
        self.gemma_agent = LlmAgent(
            name="Gemma_Worker", 
            model_type="gemma", 
            instruction="You are a helpful assistant. Keep answers short."
        )
        
        # Sub-Agent 2: Gemini (Smart)
        self.gemini_agent = LlmAgent(
            name="Gemini_Expert", 
            model_type="gemini", 
            instruction="You are an expert AI. Provide detailed explanations."
        )

    def route_and_execute(self, user_input):
        query_key = user_input.lower().strip()

        # Step A: Cache Check
        if query_key in self.cache:
            return f" [CACHE HIT]: {self.cache[query_key]}"

        # Step B: Routing Logic
        is_simple = any(word in query_key for word in SIMPLE_KEYWORDS) or len(user_input.split()) < 5
        
        if is_simple:
            print(f"📉 [ROUTING]: Low Cost -> Gemma 3:4b")
            response = self.gemma_agent.ask(user_input)
        else:
            print(f" [ROUTING]: High Logic -> Gemini 2.5 Flash")
            response = self.gemini_agent.ask(user_input)

        # Step C: Save to Cache
        self.cache[query_key] = response
        return response
