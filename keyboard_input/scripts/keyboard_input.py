#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 PAL Robotics SL.
# Released under the BSD License.
#
# Authors:
#   * Siegfried-A. Gevatter

import curses
import math

import rospy
from std_msgs.msg import Int8
from geometry_msgs.msg import Twist


class TextWindow():

    _screen = None
    _window = None
    _num_lines = None

    def __init__(self, stdscr, lines=10):
        self._screen = stdscr
        self._screen.nodelay(True)
        curses.curs_set(0)

        self._num_lines = lines

    def read_key(self):
        keycode = self._screen.getch()
        return keycode if keycode != -1 else None

    def clear(self):
        self._screen.clear()

    def write_line(self, lineno, message):
        if lineno < 0 or lineno >= self._num_lines:
            raise ValueError('lineno out of bounds')
        height, width = self._screen.getmaxyx()
        y = int((height / self._num_lines) * lineno)
        x = 3
        for text in message.split('\n'):
            text = text.ljust(width)
            self._screen.addstr(y, x, text)
            y += 1

    def refresh(self):
        self._screen.refresh()

    def beep(self):
        curses.flash()


class SimpleKeyTeleop():
    def __init__(self, interface):
        self._interface = interface
        self._pub_cmd = rospy.Publisher('key_input', Int8)

        self._hz = rospy.get_param('~hz', 50)

        self._forward_rate = rospy.get_param('~forward_rate', 0.8)
        self._backward_rate = rospy.get_param('~backward_rate', 0.5)
        self._rotation_rate = rospy.get_param('~rotation_rate', 1.0)
        self._last_pressed_time = 0

        self.key_now = 0

    def run(self):
        rate = rospy.Rate(self._hz)
        self._running = True
        self._interface.write_line(2, 'Pressed: None')
        self._interface.write_line(5, 'Use w,a,s,d keys to move, z to exit.')
        self._interface.write_line(7, '1: Agent action, 2: Train')
        while self._running:
            while True:
                keycode = self._interface.read_key()
                if keycode is None:
                    break
                self._key_pressed(keycode)
                self._publish(keycode)
            #rate.sleep()

    def _key_pressed(self, keycode):
        if keycode == ord('z'):
            self._running = False
            rospy.signal_shutdown('Bye')
        elif keycode > 128:
            keycode = 0
        else:
            self._last_pressed_time = rospy.get_time()

    def _publish(self,keycode):
        self._interface.clear()
        if not keycode == None:
            self.key_now = keycode
            try:
                self._pub_cmd.publish(self.key_now)
            except Exception:
                print("Wrong key")
        self._interface.write_line(2, 'Pressed: ' + chr(keycode))
        self._interface.write_line(5, 'Use w,a,s,d keys to move, z to exit.')
        self._interface.write_line(7, '1: Agent action, 2: Train.')
        self._interface.refresh()
        


def main(stdscr):
    rospy.init_node('key_teleop')
    app = SimpleKeyTeleop(TextWindow(stdscr))
    app.run()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass
