import argparse

class ArgParser():
    def __init__(self,isbaseline=False):
        self.parser = argparse.ArgumentParser(description=None)
        if not isbaseline:
            self._add_arguments()
        else:
            self._add_arguments_baseline()

    def parse_args(self):
        return self.parser.parse_args()

    def _add_arguments(self):
        self.parser.add_argument('--environ-string', default="jaco_table2", help='Environment id')
        self.parser.add_argument('--max-episode-count', default=1000, type=int, help='max number of training episodes') # 2000
        self.parser.add_argument('--training-index', default=None, type=int, help="training index")
        self.parser.add_argument('--seed', default=777, type=int, help='seed')
        self.parser.add_argument('__name', default="rl_controller", help='seed')
        self.parser.add_argument('__log', default="log_dir", help='seed')

    def _add_arguments_baseline(self):
        #self.parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
        self.parser.add_argument('--seed', help='RNG seed', type=int, default=0)
        #self.parser.add_argument('--num-timesteps', type=int, default=int(10500))
        self.parser.add_argument('--play', default=False, action='store_true')
        self.parser.add_argument('__name', default="rl_controller", help='name')
        self.parser.add_argument('__log', default="log_dir", help='log')