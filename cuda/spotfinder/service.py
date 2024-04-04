from __future__ import annotations

import logging
import subprocess
import time
import os
import threading
from pathlib import Path
from pprint import pformat
from typing import Iterator

import workflows.recipe
from rich.logging import RichHandler
from workflows.services.common_service import CommonService

DEFAULT_QUEUE_NAME = "per_image_analysis.gpu"

SPOTFINDER = Path("build/spotfinder")


def _setup_rich_logging(level=logging.DEBUG):
    """Setup a rich-based logging output. Using for debug running."""
    rootLogger = logging.getLogger()

    for handler in list(rootLogger.handlers):
        # We want to replace the streamhandler
        if isinstance(handler, logging.StreamHandler):
            rootLogger.handlers.remove(handler)
        # We also want to lower the output level, so pin this to the existing
        handler.setLevel(rootLogger.level)

    rootLogger.handlers.append(
        RichHandler(level=level, log_time_format="[%Y-%m-%d %H:%M:%S]")
    )


class GPUPerImageAnalysis(CommonService):
    _service_name = "GPU Per-Image-Analysis"
    _logger_name = "spotfinder.service"

    _spotfind_proc: subprocess.Popen | None = None

    def initializing(self):
        _setup_rich_logging()
        # self.log.debug("Checking Node GPU capabilities")
        # TODO: Write node sanity checks
        workflows.recipe.wrap_subscribe(
            self._transport,
            self._environment.get("queue") or DEFAULT_QUEUE_NAME,
            self.gpu_per_image_analysis,
            acknowledgement=True,
            log_extender=self.extend_log,
        )

    def gpu_per_image_analysis(
        self, rw: workflows.recipe.RecipeWrapper, header: dict, message: dict,
        base_path="/dev/shm/eiger"
    ):
        parameters = rw.recipe_step["parameters"]

        # Reject messages without the extra info
        if parameters.get("filename", "{filename}") == "{filename}":
            # We got a request, but didn't have required hyperion info
            self.log.debug(
                f"Rejecting PIA request for {parameters['dcid']}; no valid hyperion information"
            )
            # We just want to silently kill this message, as it wasn't for us
            rw.transport.ack(header)
            return

        self.log.debug(
            f"Gotten PIA request:\nHeader:\n {pformat(header)}\nPayload:\n {pformat(rw.payload)}\n"
            f"Parameters: {pformat(rw.recipe_step['parameters'])}\n"
        )

        # Do sanity checks, then launch spotfinder
        if not SPOTFINDER.is_file():
            self.log.error("Could not find spotfinder executable: %s", SPOTFINDER)
            rw.transport.nack(header)
            return

        # Otherwise, assume that this will work for now and nack the message
        rw.transport.ack(header)

        # Form the expected path for this dataset
        expected_path = f"{base_path}/{parameters['filename']}"

        # Create a pipe for comms
        read_fd, write_fd = os.pipe()

        # Now run the spotfinder
        command = [
        str(expected_path),
        "--images",
        parameters["number_of_frames"],
        "--start-index",
        parameters["start_index"],
        "--threads",
        str(40),
        "--pipe_fd",
        str(write_fd)
        ]
        self.log.info(f"Running: {SPOTFINDER} {' '.join(str(x) for x in command)}")
        start_time = time.monotonic()

        # Set the default channel for the result
        rw.set_default_channel("result")

        def pipe_output(read_fd:int)->Iterator[str]:
            """
            Generator to read from the pipe and yield the output

            Args:
                read_fd: The file descriptor for the pipe

            Yields:
                str: A line of JSON output
            """
            # Reader function
            with os.fdopen(read_fd, 'r') as pipe_data:
                # Process each line of JSON output
                for line in pipe_data:
                    line = line.strip()
                    yield line

        def read_and_send()->None:
            """
            Read from the pipe and send the output to the result queue

            This function is intended to be run in a separate thread

            Returns:
                None
            """
            # Read from the pipe and send to the result queue
            for line in pipe_output(read_fd):
                self.log.info(f"Received: {line.strip()}") # Change log level to debug?
                rw.send_to("result", line)

            self.log.info("Results finished sending")

        # Create a thread to read and send the output
        read_and_send_data = threading.Thread(target=read_and_send)

        # Run the spotfinder
        spotfind_process = subprocess.Popen(command, executable=SPOTFINDER, pass_fds=[write_fd])

        # Close the write end of the pipe (for this process)
        # SPOTFINDER will hold the write end open until it is done
        # This will allow the read end to detect the end of the output
        os.close(write_fd)

        # Start the read thread
        read_and_send_data.start()

        # Wait for the process to finish
        spotfind_process.wait()

        # Log the duration
        duration = time.monotonic() - start_time
        self.log.info(f"Analysis complete in {duration:.1f} s")

        # Wait for the read thread to finish
        read_and_send_data.join()