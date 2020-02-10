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

class Velocity(object):

    def __init__(self, min_velocity, max_velocity, num_steps):
        assert min_velocity > 0 and max_velocity > 0 and num_steps > 0
        self._min = min_velocity
        self._max = max_velocity
        self._num_steps = num_steps
        if self._num_steps > 1:
            self._step_incr = (max_velocity - min_velocity) / (self._num_steps - 1)
        else:
            # If num_steps is one, we always use the minimum velocity.
            self._step_incr = 0

    def __call__(self, value, step):
        """
        Takes a value in the range [0, 1] and the step and returns the
        velocity (usually m/s or rad/s).
        """
        if step == 0:
            return 0

        assert step > 0 and step <= self._num_steps
        max_value = self._min + self._step_incr * (step - 1)
        return value * max_value

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
            raise ValueError, 'lineno out of bounds'
        height, width = self._screen.getmaxyx()
        y = (height / self._num_lines) * lineno
        x = 10
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
        self._pub_cmd = rospy.Publisher('key_vel', Int8)

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
        self._interface.write_line(5, 'Use arrow keys to move, q to exit.')
        while self._running:
            while True:
                keycode = self._interface.read_key()
                if keycode is None:
                    break
                self._key_pressed(keycode)
                self._publish(keycode)
            rate.sleep()

    def _key_pressed(self, keycode):
        if keycode == ord('q'):
            self._running = False
            rospy.signal_shutdown('Bye')
        elif keycode > 255:
            pass
        else:
            self._last_pressed_time = rospy.get_time()

    def _publish(self,keycode):
        self._interface.clear()
        if not keycode == None:
            self.key_now = keycode
            self._pub_cmd.publish(self.key_now)
        self._interface.write_line(2, 'Pressed: ' + chr(keycode))
        self._interface.write_line(5, 'Use arrow keys to move, q to exit.')
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
