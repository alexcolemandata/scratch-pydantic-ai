from typing import Annotated

from string import ascii_letters
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIModel
from pydantic import Field, ValidationError, AfterValidator
from pydantic import BaseModel
from rich.console import Console
from rich.live import Live
from rich.table import Table
from typing_extensions import NotRequired, TypedDict
from openai import OpenAI

from pydantic_ai import Agent

HOST = "http://localhost:11434/v1/"

MODEL_NAME = "llama3.1:8b"
NUM_CHARACTERS = 10
MAX_ATTEMPTS = 5

client = OpenAI(api_key="ollama", base_url=HOST)

ollama_model = OpenAIModel(
    model_name=MODEL_NAME,
    base_url=HOST,
    api_key="ollama_api_key",
)


def is_song_starting_with_letter(song: str) -> str:
    if song[0] not in ascii_letters:
        raise ValueError(f"{song} does not start with a letter!")

    return song


class Character(BaseModel):
    name: str
    quote: Annotated[
        str,
        Field(description="Yearbook quote if they were graduating highschool."),
    ]
    occupation: Annotated[
        str, Field(description="Likely occupation if working a modern job.")
    ]
    song: Annotated[
        str,
        Field(description="Favourite 80's song, in the form {title} - {artist}"),
        AfterValidator(is_song_starting_with_letter),
    ]


character_agent = Agent(
    ollama_model,
    result_type=Character,
    retries=2,
    system_prompt="Fill in realistic but fun details about historical figures.",
)

name_flagger = Agent(
    ollama_model,
    result_type=bool,
    retries=2,
    system_prompt="True if the name you are given is the name of someone.",
)


def init_table() -> Table:
    table = Table(
        title="Historical Characters",
        caption=f"Streaming Structured responses from {MODEL_NAME}",
        width=120,
    )
    table.add_column("ID", justify="right")
    table.add_column("Name")
    table.add_column("Quote")
    table.add_column("Occupation", justify="right")
    table.add_column("Song", justify="right")
    table.add_column("Attempts", justify="center")

    return table


def main():
    characters: list[Character] = []
    table = init_table()

    with Live(table, refresh_per_second=4) as live:
        console = live.console
        console.print("Requesting data...", style="cyan")

        while len(characters) < NUM_CHARACTERS:
            names = [character.name for character in characters]
            songs = [character.song for character in characters]
            attempts = 0

            msg = "Generate details of a historical figure"
            if names:
                msg += ". Do not pick: " + ",".join(names)
            if songs:
                msg += ". Do not use these songs: " + ", ".join(
                    song.split("-")[0].strip() for song in songs
                )

            console.print(f"Generating Character {len(characters) + 1}")
            console.print(f"[italic]{msg}")

            while attempts < MAX_ATTEMPTS:
                try:
                    result = character_agent.run_sync(msg).data
                except UnexpectedModelBehavior as e:
                    attempts += 1
                    console.print(f"Attempt {attempts}: [red]{e}")
                else:
                    try:
                        is_valid_name = name_flagger.run_sync(result.name).data
                    except UnexpectedModelBehavior as e:
                        is_valid_name = str(e)
                    else:
                        if not is_valid_name:
                            attempts += 1
                            console.print(f"Invalid name generated: [red]{result.name}")
                        else:
                            break
            else:
                continue

            character = result
            characters.append(character)

            table.add_row(
                str(len(characters)),
                character.name,
                character.quote or "…",
                character.occupation or "…",
                character.song or "…",
                str(attempts),
            )
            live.update(table)


if __name__ == "__main__":
    main()
