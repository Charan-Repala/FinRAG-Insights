import google.generativeai as genai

def start_llm_chat(history=None):
    """Start a new chat session with the Gemini model.
    
    Args:
        history: Optional list of previous chat messages
    
    Returns:
        A chat session object
    """
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    chat = model.start_chat(history=history if history else [])
    return chat

def send_llm_message(chat, prompt):
    """Send a message to the LLM and get the response stream.
    
    Args:
        chat: Chat session object
        prompt: Text prompt to send
        
    Returns:
        Stream of response chunks
    """
    response = chat.send_message(prompt, stream=True)
    return response