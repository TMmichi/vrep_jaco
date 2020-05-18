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
        self._vel_cmd = rospy.Publisher('cmd_vel', Twist)

        self._hz = rospy.get_param('~hz', 10)

        self._forward_rate = rospy.get_param('~forward_rate', 0.3)
        self._backward_rate = rospy.get_param('~backward_rate', 0.3)
        self._rotation_rate = rospy.get_param('~rotation_rate', 0.2)
        self._last_pressed = {}
        self._last_pressed_time = 0
        self._angular = 0
        self._linear = 0

        self.key_now = 0

    movement_bindings = {
        curses.KEY_UP:    ( 1,  0),
        curses.KEY_DOWN:  (-1,  0),
        curses.KEY_LEFT:  ( 0,  1),
        curses.KEY_RIGHT: ( 0, -1),
    }

    def run(self):
        rate = rospy.Rate(self._hz)
        self._running = True
        self._interface.write_line(2, 'Pressed: None')
        self._interface.write_line(4, 'Position: q,e,w,s,a,d')
        self._interface.write_line(5, 'Orientation: y,i,u,j,h,k')
        self._interface.write_line(7, '7: Nav_Enable, 8: Nav_Disable')
        self._interface.write_line(8, '9: Key_Enable, 0: Key_Disable')
        self._interface.write_line(9, 'z: Exit')
        while self._running:
            while True:
                keycode = self._interface.read_key()
                if keycode is None:
                    break
                self._key_pressed(keycode)
                self._publish(keycode)
            self._set_velocity()
            self._vel_publish()
            rate.sleep()

    def _get_twist(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        return twist
    
    def _set_velocity(self):
        now = rospy.get_time()
        keys = []
        for a in self._last_pressed:
            if now - self._last_pressed[a] < 0.005:
                keys.append(a)
        linear = 0.0
        angular = 0.0
        for k in keys:
            l, a = self.movement_bindings[k]
            linear += l
            angular += a
        if linear > 0:
            self._linear += linear * self._forward_rate
        else:
            self._linear += linear * self._backward_rate
        _angular = angular * self._rotation_rate
        self._angular = angular


    def _key_pressed(self, keycode):
        if keycode == ord('z'):
            self._running = False
            rospy.signal_shutdown('Bye')
        elif keycode == ord('1'):
            self._last_pressed = {}
            self._last_pressed_time = 0
            self._angular = 0
            self._linear = 0
            self.key_now = 0
        elif keycode in self.movement_bindings:
            self._last_pressed[keycode] = rospy.get_time()
        else:
            self._last_pressed_time = rospy.get_time()

    def _publish(self,keycode):
        self._interface.clear()
        if not keycode == None:
            self.key_now = keycode
            if self.key_now not in self.movement_bindings:
                try:
                    self._pub_cmd.publish(self.key_now)
                except Exception:
                    print("Wrong key")
        self._interface.write_line(2, 'Pressed: ' + chr(keycode))
        self._interface.write_line(4, 'Position: q,e,w,s,a,d')
        self._interface.write_line(5, 'Orientation: y,i,u,j,h,k')
        self._interface.write_line(7, '7: Nav_Enable, 8: Nav_Disable')
        self._interface.write_line(8, '9: Key_Enable, 0: Key_Disable')
        self._interface.write_line(9, 'z: Exit')
        self._interface.refresh()
    
    def _vel_publish(self):
        self._interface.clear()
        self._interface.write_line(2, 'Pressed: ' + "arrow key")
        self._interface.write_line(4, 'Position: q,e,w,s,a,d')
        self._interface.write_line(5, 'Orientation: y,i,u,j,h,k')
        self._interface.write_line(7, '7: Nav_Enable, 8: Nav_Disable')
        self._interface.write_line(8, '9: Key_Enable, 0: Key_Disable')
        self._interface.write_line(9, 'z: Exit')
        self._interface.refresh()

        twist = self._get_twist(self._linear, self._angular)
        self._vel_cmd.publish(twist)


def main(stdscr):
    rospy.init_node('key_teleop')
    app = SimpleKeyTeleop(TextWindow(stdscr))
    app.run()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass
