from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
import time
import re
from functools import lru_cache
from pathlib import Path
import shutil
import requests

R = "\033[31m"
G = "\033[32m"
B = "\033[34m"
P = "\033[35m"
GRAY = "\033[37m"
BOLD = "\033[1m"
NC = "\033[0m"

DCSERVER = "https://ssx-dcserver.diamond.ac.uk"

progress = itertools.cycle(["▸▹▹▹▹", "▹▸▹▹▹", "▹▹▸▹▹", "▹▹▹▸▹", "▹▹▹▹▸"])


@lru_cache
def _get_auth_headers() -> dict[str, str]:
    try:
        TOKEN = os.environ["DCSERVER_TOKEN"]
    except KeyError:
        sys.exit(
            f"{R}{BOLD}Error: No credentials specified. Please set DCSERVER_TOKEN.{NC}"
        )

    return {"Authorization": "Bearer " + TOKEN}


@lru_cache
def is_visit(selector: str) -> bool:
    return re.match(r"^[a-z][a-z]+\d+-\d+$", selector)


def get_handle_error(*args, **kwargs):
    resp = requests.get(*args, **kwargs)
    if resp.status_code == 403:
        sys.exit(f"{R}{BOLD}Error: Unauthorised: " + resp.json()["detail"] + NC)
    resp.raise_for_status()
    return resp


class DCIDFetcher:
    highest_dcid: int | None
    visit: str

    def __init__(
        self,
        visit_or_beamline: str,
        since: int | None,
        *,
        end_of_collection: bool = False,
    ):
        """
        Fetch DCIDs from a selection criteria.

        Args:
            visit_or_beamline: Visit or beamline name
            since: The minimum DCID to select from
            end_of_collection: Only return DCID whose collection has completed (has endTime)
        """
        self.visit_or_beamline = visit_or_beamline
        self.highest_dcid = since
        self.end_of_collection = end_of_collection

    def prefetch(self) -> None:
        # Don't prefetch if we already got information
        if self.highest_dcid:
            return
        if is_visit(self.visit_or_beamline):
            resp = get_handle_error(
                f"{DCSERVER}/visit/{self.visit_or_beamline}/dc",
                headers=_get_auth_headers(),
                params=params,
            )
            self.highest_dcid = max([x["dataCollectionId"] for x in dcs])
            return dcs
        else:
            resp = get_handle_error(
                f"{DCSERVER}/beamline/{self.visit_or_beamline}/dc",
                headers=_get_auth_headers(),
                params={"most_recent": 0},
            )
            self.highest_dcid = resp.json()["highest_dcid"]
            # We don't know how many discarded here
            return []

    def fetch(self) -> list[dict]:
        params = {}
        if self.highest_dcid is not None:
            params = {"since_dcid": self.highest_dcid}

        if is_visit(self.visit_or_beamline):
            resp = get_handle_error(
                f"{DCSERVER}/visit/{self.visit_or_beamline}/dc",
                headers=_get_auth_headers(),
                params=params,
            )
            dcs = resp.json()
        else:
            resp = get_handle_error(
                f"{DCSERVER}/beamline/{self.visit_or_beamline}/dc",
                headers=_get_auth_headers(),
                params=params,
            )
            dcs = resp.json()["dcs"]

        if self.end_of_collection:
            dcs = [x for x in dcs if x.get("endTime")]

        # If we got collections, update our latest DCID
        if dcs:
            self.highest_dcid = max(
                self.highest_dcid or 0, *[x["dataCollectionId"] for x in dcs]
            )
        return dcs


def run():
    parser = argparse.ArgumentParser(
        description="Watch a visit for data collections and launch spotfinding"
    )
    parser.add_argument(
        "visit_or_beamline", help="The name of the visit or beamline to watch."
    )
    parser.add_argument(
        "command",
        help="Command and arguments to run upon finding a new data collection. Use '{}' to interpose the image name. Use '{nimages}' to interpose the image count, and '{startimagenumber}' for the starting image number.",
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "-t",
        help="Time (in seconds) to wait between requests. Default: %(default)ss",
        default=5,
        type=int,
        dest="wait",
    )
    parser.add_argument(
        "--all",
        help="Trigger on all collections, including existing.",
        action="store_true",
    )
    parser.add_argument(
        "--since", help="Ignore collections with DCIDs lower than this.", type=int
    )
    parser.add_argument(
        "--ended",
        help="Only run once a collection has ended",
        action="store_true",
    )
    args = parser.parse_args()

    # Prepare the execute function
    def _prepare_command(
        image: Path | str, nimages: str | int, startImageNumber: str | int
    ) -> list[str]:
        command = args.command[:]
        has_image = False
        for i in range(len(command)):
            if "{nimages}" in command[i]:
                command[i] = command[i].replace("{nimages}", str(nimages))
            if "{nimages}" in command[i]:
                command[i] = command[i].replace("{nimages}", str(nimages))
            if "{startimagenumber}" in command[i]:
                command[i] = command[i].replace(
                    "{startimagenumber}", str(startImageNumber)
                )
            if "{}" in command[i]:
                command[i] = command[i].replace("{}", str(image))
                has_image = True

        if not has_image:
            command.append(str(image))

        return command

    if args.command:
        if not Path(args.command[0]).is_file() and not shutil.which(args.command[0]):
            if args.command[0].startswith("-"):
                sys.exit(
                    f"Error: Got invalid command '{args.command[0]}'. This does not exist. You "
                    "might be trying to pass an option after the visit name."
                )
            sys.exit(f"Error: Command {args.command[0]} appears not to exist")

        print(
            f"Running command on data collection:{BOLD}{P}",
            shlex.join(
                _prepare_command("<filename>", "<nimages>", "<startimagenumber>")
            )
            + NC,
        )

    fetcher = DCIDFetcher(
        args.visit_or_beamline, since=args.since, end_of_collection=args.ended
    )

    if not args.all:
        existing = fetcher.prefetch()
        # We only know how many discarding if visit
        if is_visit(args.visit_or_beamline):
            # Grab all existing DCIDs first, so we don't get overloaded
            if existing:
                print(
                    f"Discarding {BOLD}{len(existing)}{NC} pre-existing data collections"
                )
            else:
                print(f"No existing data collections.")

    if is_visit(args.visit_or_beamline):
        word = "in"
    else:
        word = "on"

    print(
        f"Waiting for more data collections {word} {BOLD}{args.visit_or_beamline}{NC}...\n"
    )
    while True:
        if new_dcs := sorted(fetcher.fetch(), key=lambda x: x["dataCollectionId"]):
            for dc in new_dcs:
                print(
                    f"\rFound new datacollection: {BOLD}{dc['dataCollectionId']}{NC} ({dc['startTime'].replace('T', ' ')})"
                )
                image_path = Path(dc["imageDirectory"]) / dc["fileTemplate"]
                print(
                    f"    {BOLD}{dc['numberOfImages']}{NC} images in {B}{image_path}{NC}"
                )
                if args.command:
                    start = time.monotonic()
                    _command = _prepare_command(
                        image=image_path,
                        nimages=dc["numberOfImages"],
                        startImageNumber=dc["startImageNumber"],
                    )
                    print("+", shlex.join(_command))
                    proc = subprocess.run(_command)
                    elapsed = time.monotonic() - start
                    if proc.returncode == 0:
                        print(f"Command done after {BOLD}{elapsed:.1f}{NC} s")
                    else:
                        print(
                            f"{R}{BOLD}Command ended with error in {BOLD}{elapsed:.1f}{NC}{R} s{NC}"
                        )
                print()

            print(
                f"Waiting for more data collections {word} {BOLD}{args.visit_or_beamline}{NC}...\n"
            )

        print(f" {next(progress)}\r", end="")
        try:
            time.sleep(args.wait)
        except KeyboardInterrupt:
            # Closing while sleeping is perfectly normal
            print("        ")
            sys.exit()
