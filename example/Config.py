import os
from dotenv import load_dotenv
from llama_index.core import PromptTemplate

load_dotenv()
file = os.path.abspath(__file__)
ROOT = os.path.abspath(os.path.join(file, "../../../../../../"))

class Config:
    # Environment Variables
    AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    AZURE_OPENAI_EMBEDDING_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_KEY", "default_value")
    AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2025-01-01-preview")
    AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "black-ada-002")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "black-ada-002")

    AZURE_OPENAI_LLM_ENDPOINT = os.getenv("AZURE_OPENAI_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    AZURE_OPENAI_LLM_KEY = os.getenv("AZURE_OPENAI_LLM_KEY", "default_value")
    AZURE_OPENAI_LLM_API_VERSION = os.getenv("AZURE_OPENAI_LLM_API_VERSION", "2023-05-15")
    AZURE_OPENAI_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL", "gpt-4")
    AZURE_OPENAI_LLM_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME", "black-4")

    AZURE_OPENAI_SEARCH_ENDPOINT = os.getenv("AZURE_OPENAI_SEARCH_ENDPOINT", "test")
    UAMI_OS_CLIENT_ID = os.getenv("UAMI_OS_CLIENT_ID", "test")
    AZURE_OPENAI_EMBEDDING_DIMENSIONS = os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "test")

    DB_HOSTNAME=os.environ["DB_HOSTNAME"]
    DB_PORT=os.environ["DB_PORT"]
    DB_USERNAME=os.environ["DB_USERNAME"]
    DB_PASSWORD=os.environ["DB_PASSWORD"]
    DB_NAME=os.environ["DB_NAME"]

    ROOT = ROOT
    SAMPLE_DATA_DIR = os.path.join(ROOT,"./data/poc")

    INCIDENT_TABLE_NAME="incident_tickets"
    APP_LINKAGE_TABLE_NAME="app_linkage"
    HEALTH_STATUS_TABLE_NAME="health_status"

    INCIDENT_EMBEDDING_TABLE_NAME= INCIDENT_TABLE_NAME + "_embedding"
    # APP_LINKAGE_EMBEDDING_TABLE_NAME= APP_LINKAGE_TABLE_NAME + "_embedding"
    # HEALTH_STATUS_EMBEDDING_TABLE_NAME= HEALTH_STATUS_TABLE_NAME + "_embedding"

class Custom_Prompt:
    SEMANTIC_QA_PROMPT = PromptTemplate(
    """
    Consider the following when generating the answer:
    - Content and Metadata for each incident is within <INCIDENT> and </INCIDENT> tag.
    - The content is tied to the metadata within the same <INCIDENT> and </INCIDENT> tag.
    - Incident metadata is within the <METADATA> and </METADATA> tag. It is formatted as `key`=>`value` and separated by `::`. 
    - Incident number, assignment group, status and service impacted can be found in incident metadata section. Prioritize finding the answer in <METADATA> tag. 
    - Do not assume. If there is lack of incidents, highlight to the user.

    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query. (use most similar/closest incident as the last resort)

    Query: {query_str}
    Answer: 
    """
    )
    REACT_AGENT_PROMPT = """
    You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

    ## Tools

    You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
    This may require breaking the task into subtasks and using different tools to complete each subtask.

    You have access to the following tools:
    {tool_desc}

    ## Additional Rules
    - Use semantic_query_engine tool for task which requires semantic search such as searching for related incident problem, solutions and descriptions. Use this tool if user ask about context-aware generic problems or solutions.
    - If the query involves identifying related records based on contextual similarity, abstract meaning, or requires interpreting unstructured descriptions, route and use semantic_query_engine. 
    - For e.g. Use semantic_query_engine for query such as `what is the incident number related to problem?`, `what is the incident number for this solution?`, `What are some of the incidents related to the service`.
    - Use text_to_sql_engine tool for task which is quantitative in nature and relies on structured data retrieval with precise filtering.
    - Use both tool if need be.

    ## Follow below rules when generating a response:
    - If user is not asking about anything related to incidents, services, healthcheck, upstream and downstream application/service, return `Invalid query. This is not within the scope of this agent.`
    - Do not assume and use only data retrieve from tools.
    - When user ask about a problem about a service, provide the possible cause, potential solutions, upstream/downstream services, and the current health status of the application (if any).

    ## Output Format

    Please answer in the same language as the question and use the following format:

    ```
    Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
    Action: tool name (one of {tool_names}) if using a tool.
    Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
    ```

    Please ALWAYS start with a Thought.

    NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

    Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

    If this format is used, the tool will respond in the following format:

    ```
    Observation: tool response
    ```

    You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

    ```
    Thought: I can answer without using any more tools. I'll use the user's language to answer
    Answer: [your answer here (In the same language as the user's question)]
    ```

    ```
    Thought: I cannot answer the question with the provided tools.
    Answer: [your answer here (In the same language as the user's question)]

    ```

    ## Current Conversation

    Below is the current conversation consisting of interleaving human and assistant messages.
    """


    def _create_text_to_sql_prompt(INCIDENT_TABLE_NAME, APP_LINKAGE_TABLE_NAME, HEALTH_STATUS_TABLE_NAME):
        text_to_sql_prompt = """
        Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. You can order the results by a relevant column to return the most interesting examples in the database.

        Never query for all the columns from a specific table, only ask for a few relevant columns given the question.

        Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. You are required to use the following format, each taking one line:

        Question: Question here
        SQLQuery: SQL Query to run
        SQLResult: Result of the SQLQuery
        Answer: Final answer here

        Only use tables listed below.
        {schema}
        """
        text_to_sql_prompt += f"""
        Do consider the followings for SQLQuery creation.
        - {INCIDENT_TABLE_NAME} gives information regarding all incident related data.
        - {APP_LINKAGE_TABLE_NAME} gives information regarding application linkage. upstream application corresponds to from_service column and downstream application corresponds to to_service column
        - {HEALTH_STATUS_TABLE_NAME} gives information regarding service health status. The health_status columns only takes in 'down' and 'up' values. 'down' represent service inactive/unhealthy/failure while 'up' represent service active/health/success.
        """
        text_to_sql_prompt += """
        Question: {query_str}
        SQLQuery: 
        """
        return text_to_sql_prompt

class Mapping_Config:
    incidents_column_mapping = {
        'Number': 'incident_number',
        'Short description': 'short_description',
        'Service impacted': 'service_impacted',
        'Status': 'status',
        'Assignment group': 'assignment_group',
        'Assigned to': 'assigned_to',
        'Category': 'category',
        'Type': 'type',
        'Business impact': 'business_impact',
        'Urgency': 'urgency',
        'End user impacted': 'end_user_impacted',
        'Tags': 'tags',
        'Priority': 'priority',
        'Description': 'description',
        'Age Cluster': 'age_cluster',
        'Causal change': 'causal_change',
        'Causal component': 'causal_component',
        'Causal incident': 'causal_incident',
        'Classification': 'classification',
        'Subcategory': 'subcategory',
        'Owning IT Divison': 'owning_it_division',
        'Incident end time': 'incident_end_time',
        'Incident start time': 'incident_start_time',
        'Aged incident': 'aged_incident',
        'Reported on component': 'reported_on_component',
        'Reported on system': 'reported_on_system',
        'Operational impact': 'operational_impact',
        'Reputational impact': 'reputational_impact',
        'Financial impact': 'financial_impact',
        'Regulatory impact': 'regulatory_impact',
        'Work notes': 'work_notes',
        'Potential Master': 'potential_master'
    }
    app_linkage_column_mapping = {
        'from': 'from_service',
        'to': 'to_service'
    }
    health_status_column_mapping = {
        'service': 'service',
        'health status': 'health_status'
    }
