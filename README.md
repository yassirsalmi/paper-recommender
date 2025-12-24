# Paper Recommender Agent

## Project Overview
The Paper Recommender Agent is a research paper recommendation system that utilizes a Large Language Model (LLM) to suggest relevant papers based on user queries. It employs a ResearchAgent class to manage the interaction between the user, the LLM, and various tools for searching and processing research papers.

## Key Features
- Utilizes LLM for understanding user queries and generating responses
- Employs tools for searching, ranking, and summarizing research papers
- Implements a Plan-and-Execute Agents mechanism

## Installation
1. Clone the repository: `git clone https://github.com/yassirsalmi/paper-recommender`
2. Navigate to the project directory: `cd paper-recommender`
3. Install required dependencies: `pip install -r requirements.txt`

## Configuration
The project uses environment variables stored in a `.env` file. Required variables include:
- `LLM_MODEL`: the LLM model name
- `LLM_BASE_URL`: Base URL for the LLM 
- `LLM_API_KEY`: API key for the LLM (set to "EMPTY" for local models)
- `LLM_TEMPERATURE`: Temperature setting for the LLM

### Example of .env file
```sh
LLM_MODEL="llama8b" # the name provided with the --served-model-name flag in vLLM
LLM_BASE_URL="http://localhost:8099/v1"
LLM_API_KEY="EMPTY"
LLM_TEMPERATURE="0.6"
```

## Usage
To run the Paper Recommender Agent:
1. Ensure the `.env` file is properly configured
2. Run the main script: `python src/main.py`
3. The agent will process a default query ("can you please check for me the latest research papers about LLMs") and print the result

## Key Components
1. `ResearchAgent`: The main class managing the agent's functionality
2. `LLM`: A wrapper class around the ChatOpenAI model from LangChain
3. `HuggingFaceSearch`: A tool for searching research papers on Hugging Face

## Improvements
- Enhance the planning mechanism for more complex queries and add the validation
- Integrate additional search tools beyond Hugging Face
- Improve the summarization and ranking capabilities since they are just a mock for the moment
