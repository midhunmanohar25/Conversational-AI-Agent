AutoStream: Social-to-Lead Agentic Workflow

AutoStream is a tool that helps turn media chats into business leads. It uses intelligence to understand what people are saying and then takes action.

🚀 Features

* **Intent Identification**: AutoStream figures out if someone is just saying hello asking a question or ready to buy. It uses a system to categorize messages.

* **Knowledge Base**: AutoStream answers questions about prices and company policies. It uses a database to find the answers.

* **Dynamic Tool Execution**: AutoStream collects information from people like their name and email. Then triggers a mock API to follow up.

* **Stateful Memory**: AutoStream remembers what people said before so it can have a conversation that makes sense.

🛠️ Tech Stack

* Language: Python

* Framework: LangChain and Streamlit

* AI Model: Google Gemini

* Database: FAISS

* Embeddings: HuggingFace

📦 Installation & Setup

* Clone the Repository:

```bash

git clone <your-repo-url>

cd autostream-agent

```

* Install Dependencies:

```bash

pip install -r requirements.txt

```

* Environment Variables:

Create a `.env` file. Add your Google API Key:

```

GOOGLE_API_KEY=your_actual_api_key_here

```

* Run the Application:

```bash

streamlit run app.py

```

🏗️ Architecture Explanation

The AutoStream Lead Agent uses a special design that lets it understand what people want and then take action. It has a part that figures out what someone is trying to do and then it sends them to the right place.

The system has two parts:

* **Conversation Memory**: AutoStream remembers what people said before so it can have a conversation that makes sense.

* **Lead Info State**: AutoStream tracks what information it still needs from people like their name and email.

📱 WhatsApp Deployment (Webhooks)

To use AutoStream with WhatsApp follow these steps:

* **Provider Setup**: Create an account with the WhatsApp Business API.

* **Webhook Backend**: Build an API endpoint to receive messages.

* **Webhook Configuration**: Set your API URL as the Webhook URL for messages.

When someone sends a message your server receives it and then:

* Retrieves the persons chat history from a database.

* Passes the message through the AutoStream logic.

* Sends the response back, to the person using the WhatsApp Business API.