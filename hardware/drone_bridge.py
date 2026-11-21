"""Standalone physical drone / ESP32 hardware bridge (Optional / Isolated).

Subscribes externally as a normal WebSocket client to `ws://localhost:8000/ws`.
Unplugging hardware or failing to connect never affects the simulation.

Usage (requires `websockets` and `pyserial`):
    python hardware/drone_bridge.py --ws-url ws://localhost:8000/ws --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

logger = logging.getLogger("hardware_bridge")


async def run_bridge(ws_url: str, serial_port: str | None, baud: int) -> None:
    try:
        import websockets
    except ImportError:
        logger.error("Install `websockets` to run the hardware bridge.")
        return

    ser = None
    if serial_port:
        try:
            import serial  # type: ignore

            ser = serial.Serial(serial_port, baud, timeout=0.1)
            logger.info("Connected to serial device %s @ %d", serial_port, baud)
        except Exception as exc:
            logger.warning("Serial unavailable (%s); running in log-only mode.", exc)

    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                logger.info("Connected to simulation WS at %s", ws_url)
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "sim_tick":
                        for drone in msg.get("drones", []):
                            if drone.get("id") == "tabpfn":
                                ppm = float(drone.get("ppm", 0.0))
                                status = drone.get("status", "searching")
                                cmd = f"PPM:{ppm:.1f},STATUS:{status}\n"
                                if ser is not None:
                                    try:
                                        ser.write(cmd.encode("ascii"))
                                    except Exception:
                                        pass
        except Exception as exc:
            logger.warning("WS disconnected (%s); retrying in 2s...", exc)
            await asyncio.sleep(2.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Physical Drone / ESP32 WebSocket Bridge")
    parser.add_argument("--ws-url", default="ws://localhost:8000/ws")
    parser.add_argument("--port", default=None, help="Serial port (e.g. /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()
    asyncio.run(run_bridge(args.ws_url, args.port, args.baud))


if __name__ == "__main__":
    main()
