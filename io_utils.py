class SimPrinter:
    def __init__(self, verbosity=2):
        self.verbosity = verbosity

    def print(self, env, source, msg, verbosity):
        if self.verbosity >= verbosity:
            print("t:{:<4}] {}] {}".format(env.now, source, msg))
