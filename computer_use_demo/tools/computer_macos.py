import asyncio
import base64
import io
from enum import StrEnum
from typing import Literal, TypedDict, cast, get_args
import pyautogui
from anthropic.types.beta import BetaToolComputerUse20241022Param, BetaToolUnionParam

from .base import BaseAnthropicTool, ToolError, ToolResult

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action_20241022 = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]

Action_20250124 = (
    Action_20241022
    | Literal[
        "left_mouse_down",
        "left_mouse_up",
        "scroll",
        "hold_key",
        "wait",
        "triple_click",
    ]
)

ScrollDirection = Literal["up", "down", "left", "right"]


class Resolution(TypedDict):
    width: int
    height: int


# sizes above XGA/WXGA are not recommended
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


class BaseComputerToolMacOS:
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    MacOS-specific implementation using PyAutoGUI.
    """

    name: Literal["computer"] = "computer"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 1.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def __init__(self):
        super().__init__()

        self.width = int(pyautogui.size()[0])
        self.height = int(pyautogui.size()[1])
        self.display_num = None  # Not used on MacOS

    async def __call__(
        self,
        *,
        action: Action_20241022,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        print(
            f"### Performing action: {action}{f", text: {text}" if text else ''}{f", coordinate: {coordinate}" if coordinate else ''}"
        )
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")

            x, y = self.validate_and_get_coordinates(coordinate)

            if action == "mouse_move":
                await asyncio.to_thread(pyautogui.moveTo, x, y)
                return ToolResult(output=f"Mouse moved successfully to X={x}, Y={y}")
            elif action == "left_click_drag":
                await asyncio.to_thread(pyautogui.mouseDown)
                await asyncio.to_thread(pyautogui.moveTo, x, y)
                await asyncio.to_thread(pyautogui.mouseUp)
                return ToolResult(output="Mouse drag action completed.")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(f"text must be a string")

            if action == "key":
                # Handle key combinations and modifiers
                # Replace 'super' with 'command'
                key_sequence = text.lower().replace("super", "command").split("+")
                key_sequence = [key.strip() for key in key_sequence]
                # Map 'cmd' to 'command' for MacOS
                key_sequence = [
                    "command" if key == "cmd" else key for key in key_sequence
                ]
                # Handle special keys that pyautogui expects
                special_keys = {
                    "ctrl": "ctrl",
                    "control": "ctrl",
                    "alt": "alt",
                    "option": "alt",
                    "shift": "shift",
                    "command": "command",
                    "tab": "tab",
                    "enter": "enter",
                    "return": "enter",
                    "esc": "esc",
                    "escape": "esc",
                    "space": "space",
                    "spacebar": "space",
                    "up": "up",
                    "down": "down",
                    "left": "left",
                    "right": "right",
                }
                key_sequence = [special_keys.get(key, key) for key in key_sequence]
                await asyncio.to_thread(pyautogui.hotkey, *key_sequence)
                return ToolResult(output=f"Key combination '{text}' pressed.")
            elif action == "type":
                await asyncio.to_thread(
                    pyautogui.write, text, interval=TYPING_DELAY_MS / 1000.0
                )
                return ToolResult(output=f"Typed text: {text}")

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                x, y = pyautogui.position()
                x, y = self.scale_coordinates(ScalingSource.COMPUTER, int(x), int(y))
                return ToolResult(output=f"X={x},Y={y}")
            else:
                if action == "left_click":
                    await asyncio.to_thread(pyautogui.click, button="left")
                    return ToolResult(output="Left click performed.")
                elif action == "right_click":
                    await asyncio.to_thread(pyautogui.click, button="right")
                    return ToolResult(output="Right click performed.")
                elif action == "middle_click":
                    await asyncio.to_thread(pyautogui.click, button="middle")
                    return ToolResult(output="Middle click performed.")
                elif action == "double_click":
                    await asyncio.to_thread(pyautogui.doubleClick)
                    return ToolResult(output="Double click performed.")

        raise ToolError(f"Invalid action: {action}")

    def validate_and_get_coordinates(self, coordinate: tuple[int, int] | None = None):
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise ToolError(f"{coordinate} must be a list of length 2")
        if not all(isinstance(i, int) and i >= 0 for i in coordinate):
            raise ToolError(f"{coordinate} must be a list of non-negative integers")

        return self.scale_coordinates(ScalingSource.API, coordinate[0], coordinate[1])

    async def screenshot(self):
        """Take a screenshot of the current screen and return the base64 encoded image."""
        # Capture screenshot using PyAutoGUI
        screenshot = await asyncio.to_thread(pyautogui.screenshot)

        # Scale if needed
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        if self._scaling_enabled and (width != self.width or height != self.height):
            screenshot = screenshot.resize((width, height))

        img_buffer = io.BytesIO()
        # Save the image to an in-memory buffer
        screenshot.save(img_buffer, format="PNG", optimize=True)
        img_buffer.seek(0)
        base64_image = base64.b64encode(img_buffer.read()).decode()

        return ToolResult(base64_image=base64_image)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None
        for dimension in MAX_SCALING_TARGETS.values():
            # allow some error in the aspect ratio - not all ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                break
        if target_dimension is None:
            return x, y
        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > target_dimension["width"] or y > target_dimension["height"]:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)


class ComputerTool20241022(BaseComputerToolMacOS, BaseAnthropicTool):
    api_type: Literal["computer_20241022"] = "computer_20241022"

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}


class ComputerTool20250124(BaseComputerToolMacOS, BaseAnthropicTool):
    api_type: Literal["computer_20250124"] = "computer_20250124"

    def to_params(self):
        return cast(
            BetaToolUnionParam,
            {"name": self.name, "type": self.api_type, **self.options},
        )

    async def __call__(
        self,
        *,
        action: Action_20250124,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        scroll_direction: ScrollDirection | None = None,
        scroll_amount: int | None = None,
        duration: int | float | None = None,
        key: str | None = None,
        **kwargs,
    ):
        if action in ("left_mouse_down", "left_mouse_up"):
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action=}.")
            if action == "left_mouse_down":
                await asyncio.to_thread(pyautogui.mouseDown)
                return ToolResult(output="Left mouse button down.")
            else:
                await asyncio.to_thread(pyautogui.mouseUp)
                return ToolResult(output="Left mouse button up.")

        if action == "scroll":
            if scroll_direction is None or scroll_direction not in get_args(
                ScrollDirection
            ):
                raise ToolError(
                    f"{scroll_direction=} must be 'up', 'down', 'left', or 'right'"
                )
            if not isinstance(scroll_amount, int) or scroll_amount < 0:
                raise ToolError(f"{scroll_amount=} must be a non-negative int")

            # Move to coordinate if provided
            if coordinate is not None:
                x, y = self.validate_and_get_coordinates(coordinate)
                await asyncio.to_thread(pyautogui.moveTo, x, y)

            # Press modifier key if provided
            if key:
                await asyncio.to_thread(pyautogui.keyDown, key)

            # Perform scroll - PyAutoGUI scroll is in opposite direction convention
            scroll_map = {
                "up": scroll_amount,
                "down": -scroll_amount,
                "left": scroll_amount,  # horizontal scroll
                "right": -scroll_amount,
            }
            if scroll_direction in ("up", "down"):
                await asyncio.to_thread(pyautogui.scroll, scroll_map[scroll_direction])
            else:
                await asyncio.to_thread(pyautogui.hscroll, scroll_map[scroll_direction])

            # Release modifier key if provided
            if key:
                await asyncio.to_thread(pyautogui.keyUp, key)

            return ToolResult(output=f"Scrolled {scroll_direction} {scroll_amount} units.")

        if action in ("hold_key", "wait"):
            if duration is None or not isinstance(duration, (int, float)):
                raise ToolError(f"{duration=} must be a number")
            if duration < 0:
                raise ToolError(f"{duration=} must be non-negative")
            if duration > 100:
                raise ToolError(f"{duration=} is too long.")

            if action == "hold_key":
                if text is None:
                    raise ToolError(f"text is required for {action}")
                await asyncio.to_thread(pyautogui.keyDown, text)
                await asyncio.sleep(duration)
                await asyncio.to_thread(pyautogui.keyUp, text)
                return ToolResult(output=f"Held key '{text}' for {duration} seconds.")

            if action == "wait":
                await asyncio.sleep(duration)
                return await self.screenshot()

        if action == "triple_click":
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            # Move to coordinate if provided
            if coordinate is not None:
                x, y = self.validate_and_get_coordinates(coordinate)
                await asyncio.to_thread(pyautogui.moveTo, x, y)

            # Press modifier key if provided
            if key:
                await asyncio.to_thread(pyautogui.keyDown, key)

            await asyncio.to_thread(pyautogui.click, clicks=3)

            # Release modifier key if provided
            if key:
                await asyncio.to_thread(pyautogui.keyUp, key)

            return ToolResult(output="Triple click performed.")

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            # Move to coordinate if provided
            if coordinate is not None:
                x, y = self.validate_and_get_coordinates(coordinate)
                await asyncio.to_thread(pyautogui.moveTo, x, y)

            # Press modifier key if provided
            if key:
                await asyncio.to_thread(pyautogui.keyDown, key)

            # Perform click
            if action == "left_click":
                await asyncio.to_thread(pyautogui.click, button="left")
            elif action == "right_click":
                await asyncio.to_thread(pyautogui.click, button="right")
            elif action == "middle_click":
                await asyncio.to_thread(pyautogui.click, button="middle")
            elif action == "double_click":
                await asyncio.to_thread(pyautogui.doubleClick)

            # Release modifier key if provided
            if key:
                await asyncio.to_thread(pyautogui.keyUp, key)

            return ToolResult(output=f"{action.replace('_', ' ').title()} performed.")

        return await super().__call__(
            action=action, text=text, coordinate=coordinate, key=key, **kwargs
        )
