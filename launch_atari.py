import subprocess

from prompt_atari import prompt_game, prompt_name


game = prompt_game()
name = prompt_name(game)

cuda = input("Enter cuda device: ")
assert cuda.isdigit() and int(cuda) >= 0



if input(f"\nStart {game} on cuda:{cuda}? [Y|n] ").lower() == "n":
    print("Stopping.")

else:
    cmd = f"python src/main.py wandb.name={name} env.train.id={game}NoFrameskip-v4 common.device=cuda:{cuda}"
    subprocess.run(cmd, shell=True)
