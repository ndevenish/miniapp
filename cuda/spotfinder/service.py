from __future__ import annotations

import logging
import subprocess
import time
import os
from pathlib import Path
from pprint import pformat

import workflows.recipe
from rich.logging import RichHandler
from workflows.services.common_service import CommonService

DEFAULT_QUEUE_NAME = "per_image_analysis.gpu"

SPOTFINDER = Path(
    "/dls/science/users/mep23677/cuda/miniapp/cuda/spotfinder/_build/spotfinder"
)


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

    def write_lines_to_json(self, fd, output_file):
        # Open the output file for writing JSON data
        with open(output_file, 'w') as json_file:
            while True:
                # Read a line from the file descriptor
                line = os.read(fd, 4096).decode('utf-8').strip()
                if line == "EOF":
                    break
                # Write the line to the JSON file
                json_file.write(line + '\n')
        
        os.close(fd)

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
        #expected_path = f"/dev/shm/eiger/{parameters['filename']}"
        expected_path = f"{base_path}/{parameters['filename']}"

        # Create a pipe for comms
        read_fd, write_fd = os.pipe()

        # Now run the spotfinder
        command = [SPOTFINDER, str(expected_path), "--file_fd", str(write_fd)]
        self.log.info(f"Running: {' '.join(str(x) for x in command)}")
        start_time = time.monotonic()

        # Define the output file path
        output_file = "output.json"

        # Start a subprocess to record output data from the pipe
        subprocess.Popen(['python', '-c', f'import sys; GPUPerImageAnalysis().write_lines_to_json({read_fd}, "{output_file}")'])

        # Close the write end of the pipe
        os.close(read_fd)

        # Run the spotfinder
        _result = subprocess.run(command)

        duration = time.monotonic() - start_time
        self.log.info(f"Analysis complete in {duration:.1f} s")
