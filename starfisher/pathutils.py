#!/usr/bin/env python
# encoding: utf-8

import os


starfish_dir = os.getenv("STARFISH")


class EnterStarFishDirectory(object):
    """
    Step into a directory temporarily.
    """
    def __init__(self, dirname=None):
        self.old_dir = os.getcwd()
        if dirname is None:
            self.new_dir = starfish_dir
        else:
            self.new_dir = os.path.join(starfish_dir, dirname)

    def __enter__(self):
        os.chdir(self.new_dir)

    def __exit__(self, *args):
        os.chdir(self.old_dir)
