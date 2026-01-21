import time

class KifayatiMetrics:
    def __init__(self):
        self.data = []
        # Pricing assumptions (presentation ke liye)
        self.costs = {
            "Gemma 3:4b": 0.00001, # Managed Instance cost equivalent
            "Gemini 2.5 Flash": 0.0001 # API cost equivalent
        }

    def add_log(self, model_name, query, latency):
        cost = self.costs.get(model_name, 0)
        self.data.append({
            "Model": model_name,
            "Query": query[:20] + "...",
            "Latency (s)": latency,
            "Cost ($)": cost,
            "Timestamp": time.strftime("%H:%M:%S")
        })


metrics_tracker = KifayatiMetrics()
