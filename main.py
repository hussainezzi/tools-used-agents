import os
from dotenv import load_dotenv
from typing import cast, List
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents.tool import function_tool

# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Greetings",
            message="Hello! What can you help me with today?",
        ),
        cl.Starter(
            label="Weather",
            message="Find the weather in Karachi.",
        ),
    ]


import requests # Add this import
import os       # Ensure os is imported if not already
# --- Weather Tool using your API Key (via .env) ---
@function_tool
@cl.step(type="weather tool") # Decorator for visual step in Chainlit UI
def get_weather(location: str, unit: str = "C") -> str:
    """
    Fetch the current weather for a given location using OpenWeatherMap,
    returning a short description including temperature and conditions.
    Use 'F' for Fahrenheit, otherwise Celsius ('C') is assumed.
    """
    # Retrieve the API key securely from environment variables
    api_key = os.getenv("OPENWEATHERMAP_API_KEY") # <-- Gets your specific key
    if not api_key:
        return "Error: Weather API key is not configured."

    # Base URL for OpenWeatherMap Current Weather API (Matches your info)
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    # Map our unit ('C'/'F') to OpenWeatherMap's 'units' parameter
    if unit.upper() == "F":
        units_param = "imperial"
        unit_symbol = "F"
    else:
        units_param = "metric" # Default to Celsius
        unit_symbol = "C"

    # Parameters for the API request (Matches your info: q, appid, units)
    params = {
        "q": location,
        "appid": api_key, # <-- Using your API key here
        "units": units_param
    }

    try:
        # Make the GET request to the API
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
        data = response.json()

        # Check for API-specific errors (e.g., location not found)
        if data.get("cod") != 200:
            error_message = data.get("message", f"Could not retrieve weather for {location}.")
            return f"Error: {error_message}"

        # Extract relevant information
        city_name = data.get("name", location)
        description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]

        # Format the output string
        return f"The weather in {city_name} is {temperature}°{unit_symbol} with {description}."

    # --- Robust Error Handling ---
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        if status_code == 404:
            return f"Error: Could not find weather data for location '{location}'."
        elif status_code == 401:
            # This could mean the key is wrong OR not yet activated
            return "Error: Invalid or inactive Weather API key. Please check your key and allow activation time."
        else:
            return f"Error: HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the weather service."
    except requests.exceptions.Timeout:
        return "Error: The request to the weather service timed out."
    except requests.exceptions.RequestException as req_err:
        return f"Error: An error occurred while fetching weather: {req_err}"
    except KeyError as key_err:
        print(f"DEBUG: Received unexpected data format: {data}")
        return f"Error: Received unexpected data format from the weather service for {location}. Missing key: {key_err}"
    except Exception as e:
        print(f"Unexpected error in get_weather: {e}")
        return f"An unexpected error occurred while fetching weather."
# --- End of weather tool ---


@cl.on_chat_start
async def start():
    #Reference: https://ai.google.dev/gemini-api/docs/openai
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )
    """Set up the chat session when a user connects."""
    # Initialize an empty chat history in the session.
    cl.user_session.set("chat_history", [])

    cl.user_session.set("config", config)
    agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=model)
    agent.tools.append(get_weather)
    cl.user_session.set("agent", agent)

    await cl.Message(content="Welcome to the Panaversity AI Assistant! How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    # Send a thinking message
    msg = cl.Message(content="Thinking...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    # Retrieve the chat history from the session.
    history = cl.user_session.get("chat_history") or []
    
    # Append the user's message to the history.
    history.append({"role": "user", "content": message.content})
    

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        result = Runner.run_sync(agent, history, run_config=config)
        
        response_content = result.final_output
        
        # Update the thinking message with the actual response
        msg.content = response_content
        await msg.update()

        # Append the assistant's response to the history.
        history.append({"role": "developer", "content": response_content})
        # NOTE: Here we are appending the response to the history as a developer message.
        # This is a BUG in the agents library.
        # The expected behavior is to append the response to the history as an assistant message.
    
        # Update the session with the new history.
        cl.user_session.set("chat_history", history)
        
        # Optional: Log the interaction
        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")
        
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")