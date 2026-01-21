Reducing AI Costs via Hybrid Multi-Model Infrastructure
Overview
In the current GenAI landscape, using high-reasoning models like Gemini 2.5 for every trivial task leads to unsustainable API costs and infrastructure overhead. This project implements an Optimized AI Infrastructure using a hybrid approach. It intelligently routes traffic between Gemma 3 (4b), hosted on a private Vertex AI endpoint, and Gemini 2.5 Flash.

Core Architecture
The system acts as an Agentic Router that evaluates the complexity of incoming user queries before choosing the execution path:

Low Complexity (Greetings/Simple Queries): Routed to Gemma 3 (4b) via a Vertex AI Online Prediction Endpoint.

High Complexity (Reasoning/Coding): Routed to Gemini 2.5 Flash.

Key Features
1. Multi-Model Routing Logic
The KifayatiRouter analyses the input tokens and intent to decide which model is best suited for the task, ensuring we don't "over-spend" intelligence on simple tasks.

2. Session State & Cache Management
To ensure a smooth user experience and prevent redundant initialisations:

Object Caching: The Router and Metrics Tracker are stored in st.session_state to prevent re-instantiating the model connection on every Streamlit rerun.

Cache Resilience: The system includes a robust check (if "router" not in st.session_state) to initialise the infrastructure only once, preventing common "AttributeError" issues during session refreshes.

3. Real-time FinOps Dashboard
A built-in analytics tab tracks:

Cost Savings: Comparison between a standard Gemini-only baseline vs. the Hybrid approach.

Latency Monitoring: Real-time tracking of Gemma vs. Gemini response times to ensure performance parity.

4. Infrastructure Optimisation
Vertex AI Deployment: Leverages Google Cloud's enterprise-grade hosting for open-weight models.

Cost Control: Demonstrates the "Undeploy" workflow to halt compute billing when the agent is idle.

Tech Stack
Orchestration: Python, LangChain

Models: Gemma 3 (GCP Vertex AI), Gemini 2.5 Flash (Google AI Studio)

Frontend: Streamlit (Multi-tabbed Interface)

Cloud: Google Cloud Platform (Vertex AI)

Installation
Clone the repo:

Bash

git clone https://github.com/geeta-gwalior/Reducing-AI-Costs-via-Hybrid-Multi-Model-Infrastructure.git
Set up your .env with GOOGLE_API_KEY and ENDPOINT_ID.

Install dependencies:

Bash

pip install -r requirements.txt
Run the app:

Bash

streamlit run app.py
Developed by Geeta Kakrani, Google Developer Expert (AI)
