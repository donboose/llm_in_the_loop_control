"""
2D Renderer for the drone simulator.

Architecture:
  - One OS window, subdivided into a GRID_COLS × GRID_ROWS viewport grid.
  - Each cell renders one environment: walls (grey border), obstacles (brown
    boxes), goal (green circle approximated as a 16-sided polygon), drone
    (blue disc).
  - The renderer reads ONLY from the state snapshot dict produced by
    ParallelEnv.get_state_snapshot(). It never touches PyBullet.
  - Shapes are drawn as quads (two triangles) via a minimal GLSL program.
    Circles are drawn as many thin quads (polygon approximation) — no
    geometry shaders needed, works on OpenGL 3.3 Core.
  - Call renderer.draw(snapshot) once per sim step inside your main loop.
  - Call renderer.should_close() to check if the user closed the window.
  - Call renderer.destroy() on shutdown.
"""

import math
import numpy as np
import moderngl
from moderngl_window.conf import settings
import moderngl_window as mglw


# ── Layout constants ──────────────────────────────────────────────────────────
GRID_COLS   = 8
GRID_ROWS   = 4
WINDOW_W    = 1280
WINDOW_H    = 720
CELL_W      = WINDOW_W // GRID_COLS   # pixels per cell
CELL_H      = WINDOW_H // GRID_ROWS

# ── Colour palette (RGBA, 0–1) ────────────────────────────────────────────────
C_BG        = (0.08, 0.08, 0.10, 1.0)   # dark background
C_WALL      = (0.40, 0.40, 0.45, 1.0)   # grey walls
C_OBSTACLE  = (0.65, 0.35, 0.15, 1.0)   # brown obstacles
C_GOAL      = (0.20, 0.80, 0.30, 1.0)   # green goal circle
C_DRONE     = (0.15, 0.55, 0.95, 1.0)   # blue drone
C_DRONE_DIR = (0.90, 0.90, 0.20, 1.0)   # yellow heading indicator
C_BORDER    = (0.20, 0.20, 0.25, 1.0)   # cell separator

WORLD_SIZE  = 10.0   # must match drone_2d.py
DRONE_R     = 0.2
GOAL_R      = 0.5
WALL_T      = 0.2
CIRCLE_SEGS = 20     # polygon segments used to approximate a circle


class Renderer:
    """
    OpenGL 3.3 Core renderer.  Instantiate once, call draw() every frame.
    """

    def __init__(self, num_envs: int = 32, title: str = "Drone2D Sim"):
        self.num_envs  = num_envs
        self._alive    = True

        # ── Create GLFW window via moderngl-window ────────────────────────
        settings.WINDOW.update({
            "class":      "moderngl_window.context.glfw.Window",
            "gl_version": (3, 3),
            "title":      title,
            "size":       (WINDOW_W, WINDOW_H),
            "resizable":  False,
            "vsync":      True,
            "samples":    4,    # MSAA anti-aliasing
        })
        self._win = mglw.create_window_from_settings()
        self.ctx  = self._win.ctx
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # ── Compile shaders ───────────────────────────────────────────────
        import os
        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
        with open(os.path.join(shader_dir, "quad.vert")) as f:
            vert_src = f.read()
        with open(os.path.join(shader_dir, "quad.frag")) as f:
            frag_src = f.read()
        self._prog = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=frag_src,
        )

        # Dummy VAO — geometry is generated entirely in the vertex shader
        # from gl_VertexID, so we need a VAO but no vertex buffers.
        self._vao = self.ctx.vertex_array(self._prog, [])

    # ── Public API ────────────────────────────────────────────────────────────

    def draw(self, snapshot: list[dict]):
        """
        Render one frame.  Call this once per simulation step.

        Args:
            snapshot: output of ParallelEnv.get_state_snapshot()
        """
        if not self._alive:
            return

        # Process GLFW events (keyboard, window close, etc.)
        self._win.use()           # make context current
        self._win.render(0, 0)    # calls glfwPollEvents internally

        # Full window clear
        fb = self.ctx.screen
        fb.use()
        self.ctx.viewport = (0, 0, WINDOW_W, WINDOW_H)
        self.ctx.clear(*C_BG)

        # Draw each environment in its grid cell
        for idx in range(min(self.num_envs, GRID_COLS * GRID_ROWS)):
            col = idx % GRID_COLS
            row = idx // GRID_COLS
            # OpenGL origin is bottom-left; row 0 is drawn at the TOP visually.
            # Flip row so env 0 appears top-left.
            gl_row = (GRID_ROWS - 1) - row
            vp_x = col * CELL_W
            vp_y = gl_row * CELL_H

            self.ctx.viewport = (vp_x, vp_y, CELL_W, CELL_H)

            entry = snapshot[idx] if idx < len(snapshot) and snapshot[idx] else None
            self._draw_cell(entry)

        self._win.swap_buffers()

        # Check if user closed the window
        if self._win.is_closing:
            self._alive = False

    def should_close(self) -> bool:
        return not self._alive

    def destroy(self):
        """Clean up GL resources and close the window."""
        self._prog.release()
        self._vao.release()
        self._win.destroy()
        self._alive = False

    # ── Internal drawing helpers ──────────────────────────────────────────────

    def _draw_cell(self, entry: dict | None):
        """Draw one environment cell. entry=None draws an empty cell."""

        # Cell separator border (drawn as a thin full-cell quad underneath)
        self._draw_quad(0.0, 0.0, 1.0, 1.0, C_BORDER)

        # Arena background (slightly inset from the border)
        self._draw_quad(0.0, 0.0, 0.97, 0.97, C_BG)

        if entry is None:
            return

        ws = entry["world_size"]   # e.g. 10.0

        # ── Walls (4 grey bars) ───────────────────────────────────────────
        half = ws / 2
        t    = WALL_T
        # bottom wall
        self._draw_world_rect(0, -(half + t/2), ws + t*2, t, ws, C_WALL)
        # top wall
        self._draw_world_rect(0,  (half + t/2), ws + t*2, t, ws, C_WALL)
        # left wall
        self._draw_world_rect(-(half + t/2), 0, t, ws + t*2, ws, C_WALL)
        # right wall
        self._draw_world_rect( (half + t/2), 0, t, ws + t*2, ws, C_WALL)

        # ── Obstacles ─────────────────────────────────────────────────────
        for obs in entry.get("obstacles", []):
            self._draw_world_rect(
                obs["x"], obs["y"],
                obs["half_w"] * 2, obs["half_h"] * 2,
                ws, C_OBSTACLE,
            )

        # ── Goal (green circle) ───────────────────────────────────────────
        self._draw_world_circle(
            entry["goal_x"], entry["goal_y"],
            GOAL_R, ws, C_GOAL,
        )

        # ── Drone (blue circle) ───────────────────────────────────────────
        self._draw_world_circle(
            entry["drone_x"], entry["drone_y"],
            DRONE_R, ws, C_DRONE,
        )

        # ── Heading indicator (short yellow line as a thin rect) ──────────
        yaw = entry["drone_yaw"]
        line_len = DRONE_R * 1.8
        cx = entry["drone_x"] + math.cos(yaw) * DRONE_R
        cy = entry["drone_y"] + math.sin(yaw) * DRONE_R
        self._draw_world_rect(cx, cy, line_len, 0.04, ws, C_DRONE_DIR)

    def _world_to_ndc(self, wx: float, wy: float, world_size: float):
        """
        Convert world coordinates [-ws/2, ws/2] to NDC [-1, 1].
        We add a small margin (0.9) so the arena doesn't touch cell edges.
        """
        scale = 1.8 / world_size   # 0.9 * 2
        nx =  wx * scale
        ny =  wy * scale
        return nx, ny

    def _world_size_to_ndc(self, size: float, world_size: float) -> float:
        """Convert a world-space length to NDC half-extent."""
        return (size / world_size) * 0.9

    def _draw_world_rect(
        self, wx: float, wy: float,
        w: float, h: float,
        world_size: float,
        color: tuple,
    ):
        nx, ny   = self._world_to_ndc(wx, wy, world_size)
        nw       = self._world_size_to_ndc(w / 2, world_size)
        nh       = self._world_size_to_ndc(h / 2, world_size)
        self._draw_quad(nx, ny, nw, nh, color)

    def _draw_world_circle(
        self, wx: float, wy: float,
        radius: float, world_size: float, color: tuple,
    ):
        """
        Approximate a circle as CIRCLE_SEGS thin quads radiating from centre.
        Simple, no geometry shaders, works on GL 3.3 Core.
        """
        nx, ny = self._world_to_ndc(wx, wy, world_size)
        nr     = self._world_size_to_ndc(radius, world_size)

        for i in range(CIRCLE_SEGS):
            angle = 2 * math.pi * i / CIRCLE_SEGS
            # Each "petal" is a very thin rectangle at this angle
            cx = nx + math.cos(angle) * nr * 0.5
            cy = ny + math.sin(angle) * nr * 0.5
            # Draw a tiny square per segment — together they fill the disc
            self._draw_quad(cx, cy, nr * 0.6, nr * 0.6, color)

        # Solid centre square to fill the gaps
        self._draw_quad(nx, ny, nr * 0.7, nr * 0.7, color)

    def _draw_quad(
        self, cx: float, cy: float,
        hw: float, hh: float,
        color: tuple,
    ):
        """
        Draw a single axis-aligned rectangle.

        Args:
            cx, cy: centre in NDC [-1, 1]
            hw, hh: half-width and half-height in NDC
            color:  RGBA tuple (0–1)
        """
        self._prog["u_pos"].value   = (cx, cy)
        self._prog["u_size"].value  = (hw, hh)
        self._prog["u_color"].value = color
        # 6 vertices (2 triangles), no index buffer, no vertex buffer
        self._vao.render(moderngl.TRIANGLES, vertices=6)
