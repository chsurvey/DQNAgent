import melee
from melee_env.dconfig import DolphinConfig
from melee import enums
import time

def main():
    d = DolphinConfig()
    # Set the path to your Melee ISO file
    iso_path = "C:/Users/USER/SSBM/ssbm.iso"

    # Create a Dolphin console instance
    #console = melee.Console()
    console = melee.Console(
            path=str(d.slippi_bin_path),
            tmp_home_directory=True)
    # Connect to the console and run the ISO
    console.connect()
    console.run(iso_path=iso_path)

    # Set up controllers for CPU players
    for port in range(2):
        if len(console.controllers) < port + 1:
            console.controllers.append(None)
        controller = melee.Controller(console=console, port=port)
        console.controllers[port] = controller

        # Set controller type to STANDARD for CPU players
        console.write_controller_port(port, enums.ControllerType.STANDARD)

    # Wait for a moment to ensure the game is ready
    time.sleep(2)

    # Start the match
    console.write_game_controller_reset()
    console.write_game_start()
    console.step()

    # Select characters for CPU players
    for port in range(1, 3):
        melee.MenuHelper.choose_character(
            character=enums.Character.FOX,  # Change character if needed
            gamestate=console.step(),
            controller=console.controllers[port],
            costume=0,
            swag=False,
            cpu_level=9,  # Adjust CPU level if needed
            start=True
        )

    # Choose a stage
    melee.MenuHelper.choose_stage(
        stage=enums.Stage.FINAL_DESTINATION,  # Change stage if needed
        gamestate=console.step(),
        controller=console.controllers[1]
    )

    # Wait for a moment before starting the match
    time.sleep(2)

    # Run the match loop
    while True:
        gamestate = console.step()
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            # Continue running the match
            pass
        else:
            # Match ended or in a menu state, you can break the loop or handle accordingly
            break

    # Close the console and disconnect controllers
    console.stop()

if __name__ == "__main__":
    main()
