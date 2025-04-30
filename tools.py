from typing import Optional


def search_tool(state) -> str:  # Q: how to pass GuestState as type?
    """
    This tools searchs in the web to retrieve information related to the user query.

    Parameters
    ----------
    state : GalaState
        Langgraph State subclass. Saves the relevant information regarding user query and LLM reasoning

    Returns:
        str: Web search result

    Example:
        >>> search_tool(state={
            "gala_state": gala_state,
            "chat_history": {"user": "What is Uber"}
        })
        'The Uber you know, reimagined for business. Uber for Business is a platform for managing global rides and meals, and local deliveries, for companies of any size ...'
    """
    
    
    return None


def weather_tool(arg1: type, arg2: type, arg3: type = default_value) -> output_type:
    """
    Description.

    Parameters
    ----------
    arg1 : type
        Description
    arg2 : type
        Description
    arg3 : type
        Description

    Returns:
        type:

    Example:
        >>> ('arg1', 'arg2')
        'output'
    """
    return None


def run() -> None:
    pass


if __name__ == "__main__":
    run()
