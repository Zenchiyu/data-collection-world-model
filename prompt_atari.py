from pathlib import Path

from omegaconf import OmegaConf


GAMES = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "BankHeist",
    "BattleZone",
    "Boxing",
    "Breakout",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Hero",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MsPacman",
    "Pong",
    "PrivateEye",
    "Qbert",
    "RoadRunner",
    "Seaquest",
    "UpNDown",
]


def prompt_game():
    for i, game in enumerate(GAMES):
        print(f"{i:2d}: {game}")
    while True:
        x = input("\nEnter a number: ")
        if not x.isdigit():
            print("Invalid.")
            continue
        x = int(x)
        if x < 0 or x > 25:
            print("Invalid.")
            continue
        break
    game = GAMES[x]
    return game


def prompt_name(game):
    cfg_file = Path("config/trainer.yaml")
    cfg_name = OmegaConf.load(cfg_file).wandb.name
    suffix = cfg_name.split("-", 1)[1]
    name = f"{game}-{suffix}"
    name_ = input(f"Confirm run name by pressing Enter (or enter a new name): {name}\n")
    if name_ != "":
        name = name_
    return name

