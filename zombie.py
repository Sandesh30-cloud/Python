import zombiedice
import random

class MyZombie:
    def __init__(self, name):
        self.name = name

    def turn(self, gameState):
        # First roll is always made
        diceRollResults = zombiedice.roll()
        
        if diceRollResults is None:
            return
            
        brains = diceRollResults['brains']
        shotguns = diceRollResults['shotgun']
        
        # Keep rolling until we get 2 brains or 2 shotguns
        while diceRollResults and shotguns < 2:
            if brains >= 2:
                # Success! Stop rolling
                break
                
            diceRollResults = zombiedice.roll()
            if diceRollResults:
                brains += diceRollResults['brains']
                shotguns += diceRollResults['shotgun']

class ConservativeZombie:
    def __init__(self, name):
        self.name = name

    def turn(self, gameState):
        # Only rolls once and stops
        zombiedice.roll()

class AggressiveZombie:
    def __init__(self, name):
        self.name = name

    def turn(self, gameState):
        diceRollResults = zombiedice.roll()
        
        while diceRollResults:
            shotguns = diceRollResults['shotgun']
            
            # Stop if we get 2 or more shotguns
            if shotguns >= 2:
                break
                
            diceRollResults = zombiedice.roll()

class RandomZombie:
    def __init__(self, name):
        self.name = name

    def turn(self, gameState):
        diceRollResults = zombiedice.roll()
        
        while diceRollResults and random.randint(0, 1):
            diceRollResults = zombiedice.roll()

# Updated zombies tuple with new strategies
zombies = (
    zombiedice.examples.RandomCoinFlipZombie(name='Random'),
    zombiedice.examples.RollsUntilInTheLeadZombie(name='Until Leading'),
    zombiedice.examples.MinNumShotgunsThenStopsZombie(name='Stop at 2 Shotguns', minShotguns=2),
    zombiedice.examples.MinNumShotgunsThenStopsZombie(name='Stop at 1 Shotgun', minShotguns=1),
    MyZombie(name='Strategic Zombie'),
    ConservativeZombie(name='Conservative'),
    AggressiveZombie(name='Aggressive'),
    RandomZombie(name='Random Decision'),
)

# Uncomment one of the following lines to run in CLI or Web GUI mode:
#zombiedice.runTournament(zombies=zombies, numGames=1000)
zombiedice.runWebGui(zombies=zombies, numGames=1000)