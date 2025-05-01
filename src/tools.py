from typing import Optional
import os
from langchain_core.tools import tool
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import logging
import getpass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler("logs/tools.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

for handler in logger.handlers:
    handler.flush()
    
# TODO: Migrate _get_var to utils script
def _get_var(var) -> str:  # Cambiar el tipo de retorno a str
    if os.getenv(var):
        logger.info(f"{var} successfully processed")
        return os.getenv(var)  # Añadir return
    else:
        os.environ[var] = getpass.getpass(prompt=f"Type the value of {var}: ")
        return os.environ[var]  # Retornar el valor

# TODO: integrate with agent
@tool
def search_tool(query) -> str:  # Q: how to pass GuestState as type?
    """
    This tools searchs in the web to retrieve information related to the user query.

    Parameters
    ----------
    state : GalaState
        Langgraph State subclass. Saves the relevant information regarding user query and LLM reasoning

    Returns:
        str: Web search result

    Example:
        >>> search_tool.invoke(state={
            "gala_state": gala_state,
            "chat_history": {"user": "What is Uber"}
        })
        'The Uber you know, reimagined for business. Uber for Business is a platform for managing global rides and meals, and local deliveries, for companies of any size ...'
    """
    module_name = "Web Search Tool"
    logging.info(f"[{module_name}] Running web search...")
    query = quote_plus(query)  # format query
    url = (
        f"https://html.duckduckgo.com/html/?q={query}"  # Process user query as http url
    )
    headers = {"User-Agent": "Mozilla/5.0"}

    http_query = {"url": url, "headers": headers}

    http_response = requests.get(**http_query)
    soup = BeautifulSoup(markup=http_response.text, features="html.parser")
    results_raw = soup.find_all(name="a", class_="result__a", limit=3)
    
    logging.info(f"[{module_name}] Web search completed.")
    if not results_raw:
        return "No results found."

    else:
        result = "\n".join([r.get_text() for r in results_raw])

    return result


#@lru_cache(maxsize=32)
@tool
def weather_tool(
    location: str, unit: Optional[str] = "celsius", forecast_days: Optional[int] = 0
) -> str:
    """
    Retrieves current weather or forecast information for a specific location.

    Parameters
    ----------
    location : str
        City and country, e.g. 'Madrid, Spain' or 'New York, US'
    unit : Optional[str]
        Temperature unit: 'celsius' or 'fahrenheit' (default: 'celsius')
    forecast_days : Optional[int]
        Number of forecast days (0 for current weather, max 3) (default: 0)

    Returns
    -------
    str:
        Formatted weather information as string

    Example
    -------
    >>> get_weather('Paris, France', 'celsius')
    'Current weather in Paris:\n- Temperature: 15°C (Feels like: 14°C)\n- Conditions: Partly cloudy\n- Humidity: 65%\n- Wind: 12 km/h'
    """
    module_name = "Weather Info Tool"
    logging.info(f"[{module_name}] Validating Weather API Credentials...")
        
    api_key = _get_var("WEATHER_API_KEY")
    
    if api_key:
        logging.info(f"[{module_name}] Weather API Credential succesfully processed.")
    else:
        response = "Weather API Credential invalid\nCheck https://home.openweathermap.org/api_keys for more info."
        logging.info(f"[{module_name}] {response}")
        return response

    try:
        logging.info(f"[{module_name}] Querying current weather on {location}...")
        base_url = "https://api.openweathermap.org/data/2.5/"
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric" if unit == "celsius" else "imperial",
        }

        if forecast_days > 0:
            params["cnt"] = forecast_days * 8  # 3-hour intervals
            response = requests.get(f"{base_url}forecast", params=params)
            data = response.json()
            # Process forecast data
            return f"Weather forecast for {location}"
        else:
            response = requests.get(f"{base_url}weather", params=params)
            data = response.json()

            weather_info = {
                "location": data.get("name", location),
                "temp": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "unit": "°C" if unit == "celsius" else "°F",
                "wind_unit": "km/h" if unit == "celsius" else "mph",
            }
            logging.info(f"[{module_name}] Weather in {location} succesfuly processed.")
            return (
                f"Current weather in {weather_info['location']}:\n"
                f"- Temperature: {weather_info['temp']}{weather_info['unit']} "
                f"(Feels like: {weather_info['feels_like']}{weather_info['unit']})\n"
                f"- Conditions: {weather_info['description']}\n"
                f"- Humidity: {weather_info['humidity']}%\n"
                f"- Wind: {weather_info['wind_speed']} {weather_info['wind_unit']}"
            )

    except requests.exceptions.RequestException as e:
        return f"Error retrieving weather data: {str(e)}"
    except KeyError as e:
        return f"Error processing weather data: Missing {str(e)} in API response"


def run() -> None:
    t = int(input("Que tarea? 0=web search, 1=weather query: "))
    
    if t == 0:
        query = input("que quieres buscar?: ")
        result = search_tool.invoke(query)
    elif t == 1:
        query = input("ingresa la ciudad a consultar: ")
        result = weather_tool.invoke(query)
        
    print(result)
    return result
    #pass

if __name__ == "__main__":
    run()