"""Play a local map with the keyboard or watch the submitted DoomAgent."""

import argparse
from contextlib import closing
import sys
from pathlib import Path

import numpy as np
import pygame

from doom_env import make_env
from doom_env.spaces import FRAME_SKIP, VITALS_INDEX

ROOT = Path(__file__).resolve().parents[1]
BUTTON_KEYS = (
    pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d,
    pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_e, pygame.K_q,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wad", type=Path, default=ROOT / "maps" / "two_rooms.wad")
    parser.add_argument("--map", default="MAP01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=4200)
    parser.add_argument("--bot", action="store_true")
    parser.add_argument("--weights", type=Path, help="optional checkpoint for --bot")
    args = parser.parse_args()
    if args.weights and not args.bot:
        parser.error("--weights requires --bot")

    pygame.display.init()
    pygame.font.init()
    try:
        screen = pygame.display.set_mode((800, 600))
        with closing(make_env(args.wad, args.map, episode_ticks=args.ticks)) as env:
            agent = None
            if args.bot:
                sys.path.insert(0, str(ROOT / "submission"))
                import agent as submission_agent

                if args.weights:
                    submission_agent.WEIGHTS = args.weights.resolve()
                agent = submission_agent.DoomAgent(env.observation_space, env.action_space, "cpu")

            def restart():
                if agent:
                    agent.reset()
                    agent.rng = np.random.default_rng(args.seed)
                observation, _ = env.reset(seed=args.seed)
                return observation

            obs = restart()
            font = pygame.font.Font(None, 54)
            hint_font = pygame.font.Font(None, 30)

            clock = pygame.time.Clock()
            result = None
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs = restart()
                        result = None
                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    return
                frame = pygame.surfarray.make_surface(obs["rgb"].swapaxes(0, 1))
                screen.blit(pygame.transform.scale(frame, screen.get_size()), (0, 0))
                health = int(obs["vitals"][VITALS_INDEX["health"]])
                mode = "Bot" if agent else "Keyboard"
                status = f" | {result}" if result else ""
                if result:
                    panel = pygame.Surface((screen.get_width(), 142), pygame.SRCALPHA)
                    panel.fill((0, 0, 0, 220))
                    screen.blit(panel, (0, 220))
                    title = font.render(result, True, (255, 255, 255))
                    hint = hint_font.render("R to restart   ·   Esc to close", True, (255, 255, 255))
                    screen.blit(title, title.get_rect(center=(400, 264)))
                    screen.blit(hint, hint.get_rect(center=(400, 316)))
                pygame.display.set_caption(f"DevForge DOOM | {mode} | HP {health}{status}")
                pygame.display.flip()
                if result is None:
                    action = agent.act(obs) if agent else np.array([keys[key] for key in BUTTON_KEYS], dtype=np.int8)
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        state = env.privileged_state()
                        if state["exit_reached"]:
                            result = "Level complete"
                        elif state["dead"]:
                            result = "Player died"
                        else:
                            result = "Time limit reached"
                        print(f"Exit: {state['exit_reached']} | Kills: {state['kills']} | Ticks: {state['tic']}", flush=True)
                clock.tick(35 / FRAME_SKIP)
    finally:
        pygame.font.quit()
        pygame.display.quit()


if __name__ == "__main__":
    main()
