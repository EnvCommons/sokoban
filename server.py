from openreward.environments import Server
from env import SokobanEnvironment

if __name__ == "__main__":
    server = Server([SokobanEnvironment])
    server.run()
